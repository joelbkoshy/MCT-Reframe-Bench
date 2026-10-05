"""Fetch, verify and normalise the Cognitive-Reframing corpus into benchmark items.

Each of the 600 rows in ``reframing_dataset.csv`` becomes one item. The item
carries the situation and the thought, which are what the systems under test
see, and the human-written reframe, which no system sees. The reframes are
known examples of content-level reframing: a seeded sample of 40 calibrates the
judge, and the remainder are held out to measure detector sensitivity (RQ4).
Items are keyed by upstream row number so that labels can be released
without redistributing the text.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request

import pandas as pd

from reframebench.config import (
    DEPRESSION_TYPICAL,
    ITEMS_PATH,
    N_CALIBRATION_ITEMS,
    SCREENING_PATH,
    SEED,
    UPSTREAM,
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    UPSTREAM_REPO,
)
from reframebench.safety import screen

# Upstream labels are inconsistent in case and in form ("Overgeneralization"
# vs "overgeneralizing"). They are mapped onto one canonical name each.
_TRAP_ALIASES = {
    "overgeneralization": "overgeneralizing",
    "personalization": "personalizing",
    "comparing": "comparing and despairing",
    "none": "not distorted",
}


def normalise_trap(label: str) -> str:
    key = " ".join(str(label).strip().lower().split())
    return _TRAP_ALIASES.get(key, key)


def split_traps(field: str) -> list[str]:
    traps = [normalise_trap(t) for t in str(field).split(",") if t.strip()]
    return traps or ["not distorted"]


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(force: bool = False) -> None:
    """Download the pinned upstream files and verify them by SHA-256."""
    UPSTREAM.mkdir(parents=True, exist_ok=True)
    for name, expected in UPSTREAM_FILES.items():
        target = UPSTREAM / name
        if target.exists() and not force and _sha256(target) == expected:
            print(f"[fetch] {name}: present, hash verified")
            continue
        url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/data/{name}"
        print(f"[fetch] {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(payload)
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink()
            raise RuntimeError(f"{name}: hash mismatch (expected {expected}, got {actual})")
        tmp.replace(target)
        print(f"[fetch] {name}: {len(payload)} bytes, hash verified")


def verify() -> None:
    for name, expected in UPSTREAM_FILES.items():
        path = UPSTREAM / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python run.py fetch` first")
        if _sha256(path) != expected:
            raise RuntimeError(f"{path} does not match the pinned hash; re-run fetch --force")


def patient_block(situation: str, thought: str) -> str:
    """The only text a system under test receives about the person."""
    return (
        f"Here is what has been happening: {situation.strip()}\n\n"
        f"The thought I keep having is: {thought.strip()}"
    )


def build() -> dict:
    """Write data/items.jsonl and results/screening.json. Deterministic."""
    verify()
    frame = pd.read_csv(UPSTREAM / "reframing_dataset.csv")

    items, escalated = [], []
    for row, record in frame.iterrows():
        traps = split_traps(record["thinking_traps_addressed"])
        matched = screen(f"{record['situation']} {record['thought']}")
        item = {
            "item_id": f"R{row:03d}",
            "row": int(row),
            "situation": str(record["situation"]),
            "thought": str(record["thought"]),
            "reframe": str(record["reframe"]),
            "traps": traps,
            "primary_trap": traps[0],
            "n_traps": len(traps),
            "depression_typical": any(t in DEPRESSION_TYPICAL for t in traps),
            "screen_status": "FLAGGED" if matched else "SAFE",
        }
        item["patient_block"] = patient_block(item["situation"], item["thought"])
        items.append(item)
        if matched:
            escalated.append({"item_id": item["item_id"], "matched": matched})

    eligible_ids = sorted(i["item_id"] for i in items if i["screen_status"] == "SAFE")
    calibration = set(random.Random(SEED).sample(eligible_ids, N_CALIBRATION_ITEMS))
    for item in items:
        item["split"] = "calibration" if item["item_id"] in calibration else "heldout"

    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ITEMS_PATH.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "upstream_commit": UPSTREAM_COMMIT,
        "n_items": len(items),
        "n_escalated": len(escalated),
        "n_eligible": len(items) - len(escalated),
        "n_calibration": len(calibration),
        "calibration_ids": sorted(calibration),
        "escalated": escalated,
        "items_hash": items_hash(),
        "primary_trap_counts": pd.Series([i["primary_trap"] for i in items]).value_counts().to_dict(),
    }
    SCREENING_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREENING_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[build] {report['n_items']} items, {report['n_escalated']} escalated by the crisis screener")
    return report


def load(eligible_only: bool = True) -> list[dict]:
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"{ITEMS_PATH} missing; run `python run.py build` first")
    with ITEMS_PATH.open(encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    if eligible_only:
        items = [i for i in items if i["screen_status"] == "SAFE"]
    return items


def items_hash() -> str:
    return _sha256(ITEMS_PATH)[:16]
