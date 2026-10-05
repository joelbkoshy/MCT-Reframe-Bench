"""Tables and pre-specified tests. Reads results/*.csv, writes results/table_*.csv and summary.json.

The six contrasts are fixed in advance. Within each outcome (reframing,
process orientation, metacognitive-belief targeting) their p-values are
Holm-corrected. All contrasts are paired over the same items.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reframebench.arms import ARM_ORDER
from reframebench.config import (
    CALIBRATION_PATH,
    JUDGE_MIN_SENSITIVITY,
    JUDGE_MIN_SPECIFICITY,
    JUDGE_PATH,
    RESULTS,
    RUNS_PATH,
    SCREENING_PATH,
    VALIDATION_PATH,
)
from reframebench.stats import cohen_kappa, cramers_v, holm, mcnemar_exact, wilson_ci

CONTRASTS = [
    ("C1", "qwen-neutral", "qwen-mct", "specifying MCT, 1.5B"),
    ("C2", "qwen-mct", "qwen-mct+boundary", "stating the restructuring boundary, 1.5B"),
    ("C3", "llama3-neutral", "llama3-mct", "specifying MCT, 8B"),
    ("C4", "llama3-mct", "llama3-mct+boundary", "stating the restructuring boundary, 8B"),
    ("C5", "qwen-mct", "llama3-mct", "model size, MCT prompt"),
    ("C6", "qwen-mct+boundary", "llama3-mct+boundary", "model size, MCT + boundary prompt"),
]
OUTCOMES = ("reframing", "process_oriented", "mb_any")

# Traps with fewer primary items than this are pooled into "other" for RQ3.
MIN_TRAP_N = 20

_BOOL = ["depression_typical", "reframing", "names_process", "process_oriented",
         "mb_uncontrollability", "mb_usefulness", "mb_any", "refused"]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _ordered(arms) -> list[str]:
    present = set(arms)
    return [a for a in ARM_ORDER if a in present]


def _rate(sub: pd.DataFrame, col: str) -> dict:
    k, n = int(sub[col].sum()), len(sub)
    lo, hi = wilson_ci(k, n)
    return {f"{col}_rate": k / n if n else np.nan, f"{col}_ci_low": lo, f"{col}_ci_high": hi}


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RUNS_PATH)
    for col in _BOOL:
        df[col] = _as_bool(df[col])
    df["error"] = df["response"].astype(str).str.startswith("[GENERATION ERROR")
    if JUDGE_PATH.exists():
        judged = pd.read_csv(JUDGE_PATH)[["arm", "item_id", "judge_verdict"]]
        df = df.merge(judged, on=["arm", "item_id"], how="left")
        df["judge_reframing"] = df["judge_verdict"].map({"reframes": True, "no": False})
    return df


def by_arm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in _ordered(df["arm"]):
        sub = df[df["arm"] == arm]
        row = {"arm": arm, "n": len(sub)}
        for col in OUTCOMES:
            row.update(_rate(sub, col))
        if "judge_reframing" in sub:
            judged = sub.dropna(subset=["judge_reframing"])
            k, n = int(judged["judge_reframing"].sum()), len(judged)
            lo, hi = wilson_ci(k, n)
            row.update({"judge_n": n, "judge_reframing_rate": k / n if n else np.nan,
                        "judge_reframing_ci_low": lo, "judge_reframing_ci_high": hi})
        row.update({
            "names_process_rate": sub["names_process"].mean(),
            "mb_uncontrollability_rate": sub["mb_uncontrollability"].mean(),
            "mb_usefulness_rate": sub["mb_usefulness"].mean(),
            "any_technique_rate": (sub["primary_technique"] != "none").mean(),
            "refusal_rate": sub["refused"].mean(),
            "median_words": sub["n_words"].median(),
            "median_latency_s": sub["latency_s"].median(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def reframing_moves(df: pd.DataFrame) -> pd.DataFrame:
    moves = df["reframing_moves"].fillna("").str.get_dummies(sep=";")
    return pd.concat([df[["arm"]], moves], axis=1).groupby("arm").mean().reindex(
        _ordered(df["arm"])).reset_index()


def by_arm_trap(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="arm", columns="primary_trap", values="reframing", aggfunc="mean")
    return table.reindex(_ordered(table.index)).reset_index()


def depression_typical(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in _ordered(df["arm"]):
        for flag, label in ((True, "depression-typical"), (False, "other")):
            sub = df[(df["arm"] == arm) & (df["depression_typical"] == flag)]
            rows.append({"arm": arm, "subset": label, "n": len(sub), **_rate(sub, "reframing"),
                         "mb_any_rate": sub["mb_any"].mean()})
    return pd.DataFrame(rows)


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, a, b, label in CONTRASTS:
        row = {"contrast": code, "from": a, "to": b, "isolates": label}
        for col in OUTCOMES:
            wide = df.pivot(index="item_id", columns="arm", values=col)
            if a not in wide or b not in wide:
                row[f"{col}_p"] = np.nan
                continue
            pair = wide[[a, b]].dropna()
            xa, xb = pair[a].astype(bool).to_numpy(), pair[b].astype(bool).to_numpy()
            only_b, only_a, p = mcnemar_exact(xa, xb)
            row.update({f"{col}_n": len(pair), f"{col}_from": xa.mean(), f"{col}_to": xb.mean(),
                        f"{col}_only_to": only_b, f"{col}_only_from": only_a, f"{col}_p": p})
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in OUTCOMES:
        out[f"{col}_p_holm"] = holm(out[f"{col}_p"].tolist())
    return out


def _pooled_trap(df: pd.DataFrame) -> pd.Series:
    counts = df.drop_duplicates("item_id")["primary_trap"].value_counts()
    keep = set(counts[counts >= MIN_TRAP_N].index)
    return df["primary_trap"].where(df["primary_trap"].isin(keep), "other")


def technique_by_trap(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(trap=_pooled_trap(df))
    return (d.groupby(["arm", "trap"])["primary_technique"].value_counts(normalize=True)
            .rename("share").reset_index())


def personalization(df: pd.DataFrame) -> pd.DataFrame:
    """RQ3: is the technique offered independent of the thinking-trap category?"""
    d = df.assign(trap=_pooled_trap(df))
    rows = []
    for arm in _ordered(d["arm"]):
        sub = d[d["arm"] == arm]
        table = pd.crosstab(sub["trap"], sub["primary_technique"])
        v, p, dof = cramers_v(table.to_numpy())
        rows.append({"arm": arm, "n": len(sub), "n_traps": table.shape[0],
                     "n_techniques": table.shape[1], "cramers_v": v, "chi2_p": p, "dof": dof})
    out = pd.DataFrame(rows)
    out["chi2_p_holm"] = holm(out["chi2_p"].tolist())
    return out


def _sens_spec(label: pd.Series, flagged: pd.Series) -> dict:
    pos, neg = label == 1, label == 0
    tp, tn = int((flagged & pos).sum()), int((~flagged & neg).sum())
    sens, spec = wilson_ci(tp, int(pos.sum())), wilson_ci(tn, int(neg.sum()))
    return {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "sensitivity": tp / pos.sum() if pos.sum() else np.nan,
            "sens_ci_low": sens[0], "sens_ci_high": sens[1],
            "specificity": tn / neg.sum() if neg.sum() else np.nan,
            "spec_ci_low": spec[0], "spec_ci_high": spec[1]}


def calibration() -> pd.DataFrame | None:
    if not CALIBRATION_PATH.exists():
        return None
    c = pd.read_csv(CALIBRATION_PATH)
    c["lexical_reframing"] = _as_bool(c["lexical_reframing"])
    parsed = c[c["judge_verdict"].isin(["reframes", "no"])]
    rows = [{"detector": "lexical", **_sens_spec(c["label"], c["lexical_reframing"]), "n_unparsed": 0},
            {"detector": "judge", **_sens_spec(parsed["label"], parsed["judge_verdict"] == "reframes"),
             "n_unparsed": int(len(c) - len(parsed))}]
    out = pd.DataFrame(rows)
    out["meets_criterion"] = ((out["sensitivity"] >= JUDGE_MIN_SENSITIVITY)
                              & (out["specificity"] >= JUDGE_MIN_SPECIFICITY))
    by_kind = c.groupby("kind").apply(
        lambda g: pd.Series({"lexical_flag_rate": g["lexical_reframing"].mean(),
                             "judge_flag_rate": (g["judge_verdict"] == "reframes").mean()}),
        include_groups=False)
    by_kind.reset_index().to_csv(RESULTS / "table_calibration_by_kind.csv", index=False)
    return out


def validation() -> pd.DataFrame | None:
    """RQ4: sensitivity on held-out human reframes (all known positives)."""
    if not VALIDATION_PATH.exists():
        return None
    v = pd.read_csv(VALIDATION_PATH)
    v["lexical_reframing"] = _as_bool(v["lexical_reframing"])
    rows = []
    judged = v["judge_verdict"].isin(["reframes", "no"])
    for name, flagged in (("lexical", v["lexical_reframing"]),
                          ("judge", v.loc[judged, "judge_verdict"] == "reframes")):
        n = len(flagged)
        if n == 0:
            continue
        k = int(flagged.sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"detector": name, "n": n, "flagged": k, "sensitivity": k / n,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def judge_agreement(df: pd.DataFrame) -> dict | None:
    if "judge_reframing" not in df:
        return None
    parsed = df.dropna(subset=["judge_reframing"])
    lex, jud = parsed["reframing"].astype(int), parsed["judge_reframing"].astype(int)
    return {"n_judged": int(df["judge_verdict"].notna().sum()),
            "n_unparsed": int((df["judge_verdict"] == "unparsed").sum()),
            "agreement": float((lex == jud).mean()) if len(parsed) else float("nan"),
            "kappa": cohen_kappa(lex, jud),
            "lexical_only": int(((lex == 1) & (jud == 0)).sum()),
            "judge_only": int(((lex == 0) & (jud == 1)).sum()),
            "both": int(((lex == 1) & (jud == 1)).sum()),
            "neither": int(((lex == 0) & (jud == 0)).sum())}


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    if SCREENING_PATH.exists():
        s = json.loads(SCREENING_PATH.read_text(encoding="utf-8"))
        summary["screening"] = {k: s[k] for k in ("n_items", "n_escalated", "n_eligible",
                                                  "n_calibration", "items_hash")}

    tables: dict[str, pd.DataFrame] = {}
    cal = calibration()
    if cal is not None:
        tables["calibration"] = cal
    val = validation()
    if val is not None:
        tables["detector_validation"] = val

    if RUNS_PATH.exists():
        raw = load_runs()
        summary["generation_errors"] = int(raw["error"].sum())
        df = raw[~raw["error"]]
        tables.update({
            "by_arm": by_arm(df),
            "reframing_moves": reframing_moves(df),
            "reframing_by_arm_trap": by_arm_trap(df),
            "depression_typical": depression_typical(df),
            "contrasts": contrasts(df),
            "technique_by_trap": technique_by_trap(df),
            "personalization": personalization(df),
        })
        agreement = judge_agreement(df)
        if agreement is not None:
            summary["judge_agreement"] = agreement

    for name, table in tables.items():
        table.to_csv(RESULTS / f"table_{name}.csv", index=False)
        print(f"[analyse] wrote results/table_{name}.csv ({len(table)} rows)")
    summary.update({k: t.to_dict(orient="records") for k, t in tables.items()
                    if k in ("calibration", "detector_validation", "by_arm", "contrasts", "personalization")})
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("[analyse] wrote results/summary.json")
    return summary
