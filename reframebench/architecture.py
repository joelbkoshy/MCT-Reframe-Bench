"""Draw the pipeline architecture diagram (figures/architecture.png and .pdf).

Needs no data or models:

    python run.py architecture
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from reframebench.config import FIGURES

W = 21.0  # box width in canvas units

# name: (centre x, centre y, height, text, fill)
BOXES = {
    "corpus": (12, 50, 8, "Cognitive-Reframing data\nSharma et al. (2023), 600 rows\npinned commit + SHA-256", "#f2f2f2"),
    "items": (12, 39, 8, "Item builder\nsituation + thought -> user turn\nthinking-trap labels normalised", "white"),
    "screen": (12, 28, 8, "Crisis screen\ndeterministic, before generation\nflagged items excluded", "white"),
    "split": (12, 16, 8, "Seeded split of reframes\n40 calibration\nremainder held out", "white"),
    "prompts": (37.5, 49, 11, "System prompts\nneutral | MCT | MCT + boundary\nidentical except one paragraph\non restructuring", "white"),
    "models": (37.5, 36, 7, "Models\nQwen2.5 1.5B | Llama-3 8B", "white"),
    "ollama": (37.5, 25, 8, "Ollama, local only\n3 prompts x 2 models = 6 arms\nT = 0.3, top-p = 0.9, fixed seed", "#f2f2f2"),
    "negatives": (37.5, 12, 6, "Authored known negatives\n40 non-reframing texts", "#f2f2f2"),
    "scorers": (63, 46, 13, "Lexical scorers\ncontent reframing  [RQ1]\nprocess orientation\nmetacognitive beliefs  [RQ2]\ntechnique offered  [RQ3]", "white"),
    "judge": (63, 30, 8, "Judge (Mistral 7B)\nnot under test\nREFRAMES / NO rubric", "white"),
    "calib": (63, 17, 7, "Calibration\nsens. + spec. on 40 + 40\ncriterion fixed in advance", "white"),
    "validate": (63, 5, 6, "Held-out validation  [RQ4]\nsensitivity on reframes", "white"),
    "stats": (88, 24, 40, "Analysis\n\nRQ1, RQ2\nexact McNemar, paired\n6 fixed contrasts\nHolm-corrected\n\nRQ3\nchi-square, Cramer's V\n\nRQ4\nsensitivity, specificity\nWilson 95% CI\n\njudge vs lexical\nCohen's kappa", "white"),
    "outputs": (88, 52, 6, "Outputs\nresults/table_*.csv, summary.json\nfigures/*.png, *.pdf", "#f2f2f2"),
}

# (from, to, from anchor, to anchor, label, label offset, dashed, elbow point or None);
# an anchor such as "w@19" pins the y coordinate.
ARROWS = [
    ("corpus", "items", "s", "n", "", (0, 0), False, None),
    ("items", "screen", "s", "n", "", (0, 0), False, None),
    ("screen", "split", "s", "n", "", (0, 0), False, None),
    ("screen", "models", "e@29", "w@34", "eligible\nitems", (0, 1.8), False, None),
    ("prompts", "models", "s", "n", "", (0, 0), False, None),
    ("models", "ollama", "s", "n", "", (0, 0), False, None),
    ("ollama", "scorers", "e@26", "w@42", "replies", (0, 0), False, None),
    ("ollama", "judge", "e@24", "w@29", "", (0, 0), False, None),
    ("split", "calib", "e@17", "w@17", "40 reframes", (-12, 0), False, None),
    ("negatives", "calib", "e", "w@15", "", (0, 0), False, None),
    ("split", "validate", "s", "w@5", "held-out reframes", (-12, 0), True, (12, 5)),
    ("judge", "calib", "s", "n", "", (0, 0), True, None),
    ("scorers", "stats", "e@42", "w@42", "", (0, 0), False, None),
    ("judge", "stats", "e", "w@30", "", (0, 0), False, None),
    ("calib", "stats", "e", "w@17", "", (0, 0), False, None),
    ("validate", "stats", "e", "w@5", "", (0, 0), False, None),
    ("stats", "outputs", "n", "s", "", (0, 0), False, None),
]

LANES = [(12, "1  Data"), (37.5, "2  Systems under test"), (63, "3  Scoring"), (88, "4  Analysis")]


def _anchor(name: str, spec: str) -> tuple[float, float]:
    side, _, pinned = spec.partition("@")
    x, y, h, *_ = BOXES[name]
    point = {"n": (x, y + h / 2), "s": (x, y - h / 2),
             "e": (x + W / 2, y), "w": (x - W / 2, y)}[side]
    return (point[0], float(pinned)) if pinned else point


def draw() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.set_xlim(-1, 100)
    ax.set_ylim(0.5, 62)
    ax.axis("off")

    for x, label in LANES:
        ax.add_patch(FancyBboxPatch((x - W / 2 - 1.2, 1), W + 2.4, 57,
                                    boxstyle="round,pad=0,rounding_size=1.2", facecolor="none",
                                    edgecolor="#bfbfbf", linewidth=0.6, linestyle=(0, (3, 2))))
        ax.text(x, 60, label, ha="center", va="center", fontsize=9, fontweight="bold")

    for x, y, h, text, fill in BOXES.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - h / 2), W, h,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=fill, edgecolor="black", linewidth=0.8, zorder=2))
        title, _, body = text.partition("\n")
        ax.text(x, y + h / 2 - 1.0, title, ha="center", va="top", fontsize=6.6,
                fontweight="bold", zorder=3)
        ax.text(x, y + h / 2 - 2.8, body, ha="center", va="top", fontsize=5.8,
                linespacing=1.3, zorder=3)

    for src, dst, s_spec, d_spec, label, (dx, dy), dashed, elbow in ARROWS:
        start, end = _anchor(src, s_spec), _anchor(dst, d_spec)
        style = (0, (3, 2)) if dashed else "-"
        if elbow:
            ax.plot([start[0], elbow[0]], [start[1], elbow[1]], color="black", lw=0.8,
                    linestyle=style, zorder=1)
            start = elbow
        ax.annotate("", xy=end, xytext=start, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, linestyle=style,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        if label:
            ax.text((start[0] + end[0]) / 2 + dx, (start[1] + end[1]) / 2 + dy, label,
                    ha="center", va="center", fontsize=5.6, style="italic", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"architecture.{ext}", dpi=600, bbox_inches="tight",
                    pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("[architecture] wrote figures/architecture.png and .pdf")


if __name__ == "__main__":
    draw()
