"""MCT-Reframe-Bench command line.

    python run.py fetch          download the pinned corpus and verify hashes
    python run.py build          build data/items.jsonl, crisis screen, calibration split
    python run.py calibrate      rate 40 known positives + 40 known negatives (judge check)
    python run.py validate       rate the held-out human reframes with both detectors
    python run.py generate       generate replies for every arm x item (resumable)
    python run.py judge          adjudicate every reply with the judge model (resumable)
    python run.py analyse        write results/table_*.csv and results/summary.json
    python run.py figures        draw figures/fig*.png|pdf from the tables
    python run.py architecture   draw figures/architecture.png|pdf
    python run.py status         show progress of each stage
    python run.py all            every stage in order
"""

from __future__ import annotations

import argparse
import csv
import sys

from reframebench.config import (
    CALIBRATION_PATH,
    JUDGE_PATH,
    LOCK_PATH,
    N_CALIBRATION_ITEMS,
    RUNS_PATH,
    VALIDATION_PATH,
)


def _rows(path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def status() -> None:
    from reframebench import data
    from reframebench.arms import ARMS
    from reframebench.calibration import NEGATIVES

    try:
        n_items = len(data.load())
    except FileNotFoundError:
        print("items: not built (run `python run.py fetch` then `build`)")
        return
    print(f"eligible items : {n_items}")
    print(f"calibration    : {_rows(CALIBRATION_PATH)} / {N_CALIBRATION_ITEMS + len(NEGATIVES)}")
    print(f"validation     : {_rows(VALIDATION_PATH)} / {n_items - N_CALIBRATION_ITEMS}")
    print(f"generations    : {_rows(RUNS_PATH)} / {n_items * len(ARMS)}")
    print(f"judgements     : {_rows(JUDGE_PATH)} / {_rows(RUNS_PATH)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    sub.add_parser("build")
    sub.add_parser("calibrate")
    p = sub.add_parser("generate")
    p.add_argument("--arms", nargs="+", help="subset of arms (default: all six)")
    p.add_argument("--limit", type=int, help="first N eligible items only (pilot runs)")
    sub.add_parser("judge")
    p = sub.add_parser("validate")
    p.add_argument("--no-judge", action="store_true", help="lexical detector only")
    p.add_argument("--limit", type=int)
    sub.add_parser("analyse")
    sub.add_parser("figures")
    sub.add_parser("architecture")
    sub.add_parser("status")
    p = sub.add_parser("all")
    p.add_argument("--limit", type=int)

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        from reframebench.data import fetch
        fetch(force=args.force)
    elif args.cmd == "build":
        from reframebench.data import build
        build()
    elif args.cmd in ("calibrate", "generate", "judge", "validate", "all"):
        from reframebench import pipeline
        from reframebench.io_utils import single_instance

        with single_instance(LOCK_PATH):
            if args.cmd == "calibrate":
                pipeline.run_calibration()
            elif args.cmd == "generate":
                pipeline.run_generation(args.arms, args.limit)
            elif args.cmd == "judge":
                pipeline.run_judge()
            elif args.cmd == "validate":
                pipeline.run_validation(use_judge=not args.no_judge, limit=args.limit)
            else:
                from reframebench import analysis, architecture, data, figures
                data.fetch()
                data.build()
                pipeline.run_calibration()
                pipeline.run_validation(limit=args.limit)
                pipeline.run_generation(limit=args.limit)
                pipeline.run_judge()
                analysis.run()
                figures.main()
                architecture.draw()
    elif args.cmd == "analyse":
        from reframebench.analysis import run
        run()
    elif args.cmd == "figures":
        from reframebench.figures import main as figures_main
        figures_main()
    elif args.cmd == "architecture":
        from reframebench.architecture import draw
        draw()
    elif args.cmd == "status":
        status()


if __name__ == "__main__":
    sys.exit(main())
