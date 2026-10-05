"""Paths and pinned constants shared by every stage."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPSTREAM = DATA / "upstream"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ITEMS_PATH = DATA / "items.jsonl"
SCREENING_PATH = RESULTS / "screening.json"
CALIBRATION_PATH = RESULTS / "calibration.csv"
VALIDATION_PATH = RESULTS / "validation.csv"
RUNS_PATH = RESULTS / "runs.csv"
JUDGE_PATH = RESULTS / "judgements.csv"
MANIFEST_PATH = RESULTS / "run_manifest.json"
LOCK_PATH = RESULTS / ".run.lock"

# Cognitive-Reframing (Sharma et al., ACL 2023), pinned and hash-verified.
# CC BY-NC-ND 4.0: downloaded at run time, never committed.
UPSTREAM_REPO = "behavioral-data/Cognitive-Reframing"
UPSTREAM_COMMIT = "4c1d4afdc22bc66136d14d8131ee0aa4a2d1e55d"
UPSTREAM_FILES = {
    "reframing_dataset.csv": "fad43e566de3d46c62f9f9045ad6a7d843877f6e4151ec884d03e85188861209",
    "thinking_traps.jsonl": "c673e193063f0e5c5dd9cebbf7f5db33b38332d12d347cba466a2c717503733f",
}

SEED = 20261005

# Items whose human reframes calibrate the judge; the rest are held out for RQ4.
N_CALIBRATION_ITEMS = 40

# A-priori reporting subset for RQ3, fixed before any generation.
DEPRESSION_TYPICAL = (
    "all-or-nothing thinking",
    "disqualifying the positive",
    "labeling",
    "overgeneralizing",
)

# Pre-specified acceptance criterion for using the judge as a primary measure.
JUDGE_MIN_SENSITIVITY = 0.80
JUDGE_MIN_SPECIFICITY = 0.80
