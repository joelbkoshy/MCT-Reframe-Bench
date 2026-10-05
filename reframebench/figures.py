"""Publication figures, drawn from results/table_*.csv only.

Greyscale with hatching (survives monochrome print), no titles or captions in
the image, 600 dpi PNG plus vector PDF. Re-run the analysis, then this module,
and every figure is regenerated from the current results.

    python run.py figures
    python -m reframebench.figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reframebench.arms import ARM_ORDER
from reframebench.config import FIGURES, RESULTS

DPI = 600
GREYS = ["white", "#d9d9d9", "#a6a6a6", "#595959"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.edgecolor": "black",
    "hatch.linewidth": 0.6,
    "hatch.color": "#444444",
})


def _table(name: str) -> pd.DataFrame | None:
    path = RESULTS / f"table_{name}.csv"
    if not path.exists():
        print(f"[figures] skip: {path.name} not found (run `python run.py analyse`)")
        return None
    return pd.read_csv(path)


def _save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight",
                    pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote figures/{stem}.png and .pdf")


def _by_arm(frame: pd.DataFrame) -> pd.DataFrame:
    order = [a for a in ARM_ORDER if a in set(frame["arm"])]
    return frame.set_index("arm").loc[order]


def _err(values, low, high):
    return np.clip(np.vstack([values - low, high - values]), 0, None)


def _style(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.spines[["top", "right"]].set_visible(False)


def fig_reframing_by_arm() -> None:
    """RQ1: content-reframing rate per arm, lexical and judge, Wilson 95% CI."""
    t = _table("by_arm")
    if t is None:
        return
    d = _by_arm(t)
    x = np.arange(len(d))
    has_judge = "judge_reframing_rate" in d
    w = 0.38 if has_judge else 0.6
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    series = [("lexical", "reframing_rate", "reframing_ci_low", "reframing_ci_high", "white", "///")]
    if has_judge:
        series.append(("judge", "judge_reframing_rate", "judge_reframing_ci_low",
                       "judge_reframing_ci_high", "#a6a6a6", ""))
    for k, (label, col, lo, hi, colour, hatch) in enumerate(series):
        offset = (k - (len(series) - 1) / 2) * w
        v = d[col].to_numpy(float)
        ax.bar(x + offset, v, w, facecolor=colour, edgecolor="black", hatch=hatch,
               linewidth=0.8, label=label, zorder=2)
        ax.errorbar(x + offset, v, yerr=_err(v, d[lo].to_numpy(float), d[hi].to_numpy(float)),
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=3)
    _style(ax, d.index)
    ax.set_ylabel("content-reframing rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _save(fig, "fig1_reframing_by_arm")


def fig_reframing_by_trap() -> None:
    """RQ1 x RQ3: lexical reframing rate per arm and primary thinking trap."""
    t = _table("reframing_by_arm_trap")
    if t is None:
        return
    d = _by_arm(t)
    values = d.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    im = ax.imshow(values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels(d.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(d.shape[0]))
    ax.set_yticklabels(d.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if values[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, label="content-reframing rate")
    _save(fig, "fig2_reframing_by_trap")


def fig_metacognitive_beliefs() -> None:
    """RQ2: share of replies addressing each class of metacognitive belief."""
    t = _table("by_arm")
    if t is None:
        return
    d = _by_arm(t)
    x, w = np.arange(len(d)), 0.38
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(x - w / 2, d["mb_uncontrollability_rate"], w, facecolor="white", edgecolor="black",
           hatch="///", label="uncontrollability", zorder=2)
    ax.bar(x + w / 2, d["mb_usefulness_rate"], w, facecolor="#a6a6a6", edgecolor="black",
           label="usefulness of thinking", zorder=2)
    _style(ax, d.index)
    ax.set_ylabel("share of replies")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, "fig3_metacognitive_beliefs")


def fig_technique_by_trap() -> None:
    """RQ3: technique offered by thinking trap, one panel per arm."""
    t = _table("technique_by_trap")
    if t is None:
        return
    arms = [a for a in ARM_ORDER if a in set(t["arm"])]
    techniques = ["postponement", "attention", "detached_mindfulness", "none"]
    cols = 3
    rows = int(np.ceil(len(arms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.4 * rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, arm in zip(axes.flat, arms):
        sub = (t[t["arm"] == arm].pivot(index="trap", columns="primary_technique", values="share")
               .reindex(columns=techniques).fillna(0.0))
        left = np.zeros(len(sub))
        for tech, colour, hatch in zip(techniques, GREYS, ["", "///", "", ""]):
            ax.barh(sub.index, sub[tech], left=left, color=colour, edgecolor="black",
                    linewidth=0.4, hatch=hatch, label=tech.replace("_", " "))
            left += sub[tech].to_numpy()
        ax.set_title(arm, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlim(0, 1)
    for ax in list(axes.flat)[len(arms):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.03))
    fig.supxlabel("share of replies", fontsize=8, y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig4_technique_by_trap")


def fig_process_vs_reframing() -> None:
    """Arm-level process orientation against content reframing."""
    t = _table("by_arm")
    if t is None:
        return
    d = _by_arm(t)
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    for (arm, r), m in zip(d.iterrows(), ["o", "s", "^", "D", "v", "P"]):
        filled = "#595959" if arm.startswith("llama3") else "white"
        ax.scatter(r["process_oriented_rate"], r["reframing_rate"], marker=m, s=32,
                   facecolor=filled, edgecolor="black", zorder=3)
        ax.annotate(arm, (r["process_oriented_rate"], r["reframing_rate"]), fontsize=6.3,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("process-oriented replies")
    ax.set_ylabel("content-reframing replies")
    ax.grid(alpha=0.3, linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig5_process_vs_reframing")


def fig_detector_validation() -> None:
    """RQ4: calibration sensitivity/specificity and held-out sensitivity, per detector."""
    cal, val = _table("calibration"), _table("detector_validation")
    if cal is None and val is None:
        return
    bars = []
    if cal is not None:
        for _, r in cal.iterrows():
            bars.append((f"{r['detector']}\ncal. sens.", r["sensitivity"], r["sens_ci_low"], r["sens_ci_high"], "white", "///"))
            bars.append((f"{r['detector']}\ncal. spec.", r["specificity"], r["spec_ci_low"], r["spec_ci_high"], "#d9d9d9", ""))
    if val is not None:
        for _, r in val.iterrows():
            bars.append((f"{r['detector']}\nheld-out sens.", r["sensitivity"], r["ci_low"], r["ci_high"], "#a6a6a6", ""))
    labels, v, lo, hi, colours, hatches = map(list, zip(*bars))
    v, lo, hi = (np.array(a, float) for a in (v, lo, hi))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(bars))
    ax.bar(x, v, 0.62, color=colours, edgecolor="black", hatch=None, linewidth=0.8, zorder=2)
    for patch, hatch in zip(ax.patches, hatches):
        patch.set_hatch(hatch)
    ax.errorbar(x, v, yerr=_err(v, lo, hi), fmt="none", ecolor="black", elinewidth=0.8,
                capsize=2.5, zorder=3)
    ax.axhline(0.8, color="black", linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("proportion")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig6_detector_validation")


def main() -> None:
    fig_reframing_by_arm()
    fig_reframing_by_trap()
    fig_metacognitive_beliefs()
    fig_technique_by_trap()
    fig_process_vs_reframing()
    fig_detector_validation()


if __name__ == "__main__":
    main()
