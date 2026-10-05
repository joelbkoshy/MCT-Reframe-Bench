"""Calibration, validation, generation and adjudication. Every stage is resumable."""

from __future__ import annotations

import csv
import json

from reframebench import data
from reframebench.arms import ARM_BY_NAME, ARMS
from reframebench.backend import NUM_PREDICT, TEMPERATURE, TOP_P, generate, installed_models
from reframebench.calibration import NEGATIVES
from reframebench.config import (
    CALIBRATION_PATH,
    JUDGE_PATH,
    MANIFEST_PATH,
    RUNS_PATH,
    SEED,
    VALIDATION_PATH,
)
from reframebench.io_utils import ResumableWriter, progress
from reframebench.judge import JUDGE_MODEL, adjudicate
from reframebench.scoring import detect_reframing, score

RUN_FIELDS = [
    "arm", "model", "size", "prompt",
    "item_id", "split", "primary_trap", "traps", "depression_typical",
    "reframing", "reframing_moves", "reframing_hits",
    "names_process", "process_oriented",
    "mb_uncontrollability", "mb_usefulness", "mb_any", "mb_hits",
    "techniques", "primary_technique", "refused", "n_words", "latency_s", "response",
]
JUDGE_FIELDS = ["arm", "item_id", "lexical_reframing", "judge_verdict", "judge_rationale"]
CALIBRATION_FIELDS = ["case_id", "kind", "label", "item_id", "lexical_reframing",
                      "judge_verdict", "judge_rationale"]
VALIDATION_FIELDS = ["item_id", "primary_trap", "lexical_reframing", "reframing_moves",
                     "judge_verdict", "judge_rationale"]


def _require(models: set[str]) -> None:
    missing = models - installed_models()
    if missing:
        raise RuntimeError(f"models not installed in Ollama: {sorted(missing)} (see README, Setup)")


def _lexical(text: str) -> tuple[bool, str]:
    moves = detect_reframing(text)
    return any(moves.values()), ";".join(k for k, v in moves.items() if v)


def run_calibration() -> None:
    """Rate 40 known positives and 40 known negatives with both detectors."""
    _require({JUDGE_MODEL})
    items = sorted((i for i in data.load() if i["split"] == "calibration"), key=lambda i: i["item_id"])
    cases = [(f"pos-{i['item_id']}", "human_reframe", 1, i, i["reframe"]) for i in items]
    cases += [(f"neg-{n:02d}", kind, 0, items[n % len(items)], text)
              for n, (kind, text) in enumerate(NEGATIVES)]

    with ResumableWriter(CALIBRATION_PATH, CALIBRATION_FIELDS, key=("case_id",)) as writer:
        for done, (case_id, kind, label, item, text) in enumerate(cases, 1):
            if writer.already_done(case_id):
                continue
            lexical, _ = _lexical(text)
            verdict, reason = adjudicate(item["situation"], item["thought"], text)
            writer.write({"case_id": case_id, "kind": kind, "label": label,
                          "item_id": item["item_id"], "lexical_reframing": lexical,
                          "judge_verdict": verdict, "judge_rationale": reason})
            progress("calibrate", done, len(cases), f"{case_id} lexical={lexical} judge={verdict}")


def run_validation(use_judge: bool = True, limit: int | None = None) -> None:
    """RQ4: rate the held-out human reframes, all known positives, with both detectors."""
    if use_judge:
        _require({JUDGE_MODEL})
    items = [i for i in data.load() if i["split"] == "heldout"]
    items = items[:limit] if limit else items
    with ResumableWriter(VALIDATION_PATH, VALIDATION_FIELDS, key=("item_id",)) as writer:
        for done, item in enumerate(items, 1):
            if writer.already_done(item["item_id"]):
                continue
            lexical, moves = _lexical(item["reframe"])
            verdict, reason = (adjudicate(item["situation"], item["thought"], item["reframe"])
                               if use_judge else ("not_run", ""))
            writer.write({"item_id": item["item_id"], "primary_trap": item["primary_trap"],
                          "lexical_reframing": lexical, "reframing_moves": moves,
                          "judge_verdict": verdict, "judge_rationale": reason})
            progress("validate", done, len(items), f"{item['item_id']} lexical={lexical} judge={verdict}")


def _check_manifest() -> None:
    """Refuse to append to results produced from different items or decoding."""
    current = {"items_hash": data.items_hash(),
               "decoding": {"temperature": TEMPERATURE, "top_p": TOP_P,
                            "num_predict": NUM_PREDICT, "seed": SEED}}
    if MANIFEST_PATH.exists():
        recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if recorded != current:
            raise RuntimeError(f"items or decoding changed since this run began "
                               f"(recorded {recorded}, current {current}); move results/ aside")
    else:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")


def run_generation(arm_names: list[str] | None = None, limit: int | None = None) -> None:
    _check_manifest()
    unknown = set(arm_names or []) - set(ARM_BY_NAME)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {list(ARM_BY_NAME)}")
    arms = [ARM_BY_NAME[n] for n in arm_names] if arm_names else list(ARMS)
    items = data.load()
    items = items[:limit] if limit else items
    _require({a.model for a in arms})

    total, done = len(arms) * len(items), 0
    with ResumableWriter(RUNS_PATH, RUN_FIELDS, key=("arm", "item_id")) as writer:
        # Item-major order keeps every arm at the same coverage if a run is stopped early.
        for item in items:
            for arm in arms:
                done += 1
                if writer.already_done(arm.name, item["item_id"]):
                    continue
                try:
                    text, latency = generate(arm.system_prompt, item["patient_block"], arm.model)
                except Exception as exc:
                    text, latency = f"[GENERATION ERROR: {exc}]", 0.0
                writer.write({
                    "arm": arm.name, "model": arm.model, "size": arm.size, "prompt": arm.prompt,
                    "item_id": item["item_id"], "split": item["split"],
                    "primary_trap": item["primary_trap"], "traps": ";".join(item["traps"]),
                    "depression_typical": item["depression_typical"],
                    **score(text), "latency_s": round(latency, 2), "response": text,
                })
                progress("generate", done, total, f"{arm.name} {item['item_id']} {latency:.1f}s")


def run_judge() -> None:
    """Adjudicate every generated reply for content reframing."""
    _require({JUDGE_MODEL})
    by_id = {i["item_id"]: i for i in data.load()}
    with RUNS_PATH.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if not r["response"].startswith("[GENERATION ERROR")]
    with ResumableWriter(JUDGE_PATH, JUDGE_FIELDS, key=("arm", "item_id")) as writer:
        for done, row in enumerate(rows, 1):
            if writer.already_done(row["arm"], row["item_id"]):
                continue
            item = by_id[row["item_id"]]
            verdict, reason = adjudicate(item["situation"], item["thought"], row["response"])
            writer.write({"arm": row["arm"], "item_id": row["item_id"],
                          "lexical_reframing": row["reframing"],
                          "judge_verdict": verdict, "judge_rationale": reason})
            progress("judge", done, len(rows), f"{row['arm']} {row['item_id']} -> {verdict}")
