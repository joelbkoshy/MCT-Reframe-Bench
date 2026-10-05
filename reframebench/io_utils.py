"""Resumable, crash-safe CSV writing and a single-instance lock.

Rows are appended as they are produced and finished cells are skipped on
restart. A process killed mid-write can leave a truncated last record (for
example an unclosed quote in the long ``response`` field) whose key fields
would otherwise mark the cell as done, so every file is re-parsed and
rewritten before it is appended to.
"""

from __future__ import annotations

import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path

csv.field_size_limit(10_000_000)


@contextmanager
def single_instance(lock_path: Path):
    """Refuse to start if another run holds the lock (OS-level, released on death)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    handle.seek(0)  # lock byte 0 explicitly; append mode starts at EOF
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            print(f"another run is already in progress (lock: {lock_path})", file=sys.stderr)
            raise SystemExit(2)
        yield
    finally:
        try:
            handle.close()
        except Exception:
            pass


class ResumableWriter:
    def __init__(self, path: Path, fieldnames: list[str], key: tuple[str, ...]):
        self.path = path
        self.fieldnames = fieldnames
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair()
        self._done = self._load_done()
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._fh.flush()

    def _repair(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        ends_clean = self.path.read_bytes().endswith(b"\n")
        with self.path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except (StopIteration, csv.Error):
                return
            if header != self.fieldnames:
                raise RuntimeError(
                    f"{self.path.name} was written with a different schema; move it aside."
                )
            complete, dropped = [], 0
            try:
                for row in reader:
                    if len(row) == len(header):
                        complete.append(row)
                    else:
                        dropped += 1
            except csv.Error:
                dropped += 1
        if not ends_clean and complete:
            complete.pop()
            dropped += 1
        tmp = self.path.with_suffix(self.path.suffix + ".repair")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(complete)
        tmp.replace(self.path)
        if dropped:
            print(f"[resume] repaired {self.path.name}: dropped {dropped}, kept {len(complete)}",
                  flush=True)

    def _load_done(self) -> set[tuple]:
        if not self.path.exists():
            return set()
        with self.path.open(newline="", encoding="utf-8") as fh:
            return {tuple(row[k] for k in self.key) for row in csv.DictReader(fh)
                    if all(row.get(k) for k in self.key)}

    def already_done(self, *values) -> bool:
        return tuple(str(v) for v in values) in self._done

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()
        self._done.add(tuple(str(row[k]) for k in self.key))

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def progress(label: str, done: int, total: int, extra: str = "") -> None:
    pct = 100.0 * done / total if total else 0.0
    print(f"[{label}] {done}/{total} ({pct:5.1f}%) {extra}", flush=True)
