#!/usr/bin/env python3
"""Reproducibility checks on exact counts (not statistics).

1. Same arguments twice -> identical counts.
2. Chunked kernel with a single chunk (chunk >= n) == v0 kernel, because
   chunk 0 is seeded exactly like the original stream.
3. --compare BIN [--compare-launcher CMD]: another version (omp, mpi, ...)
   must give counts identical to the reference binary for several chunk sizes.
   This is the test every parallel version has to pass.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = "a,the,love,romeo"


def counts(cmd: list[str]) -> dict[str, int]:
    out = subprocess.run(cmd + ["-H"], check=True, capture_output=True,
                         text=True).stdout
    return {r["target"]: int(r["count"]) for r in csv.DictReader(io.StringIO(out))}


def check(name: str, a: dict, b: dict) -> bool:
    ok = a == b
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     {a}\n  != {b}"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=ROOT / "bin" / "serial")
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--compare-launcher", default="")
    ap.add_argument("--compare-args", default="",
                    help="extra options for the compared binary, e.g. '-k fused'")
    ap.add_argument("-n", default="50000000")
    args = ap.parse_args()

    ref = [str(args.bin), "-n", args.n, "-t", TARGETS, "-s", "7"]
    ok = True
    ok &= check("same run twice gives identical counts",
                counts(ref + ["-c", "2^20"]), counts(ref + ["-c", "2^20"]))
    ok &= check("one chunk (chunk >= n) equals the v0 single stream",
                counts(ref + ["-c", args.n]), counts(ref + ["-k", "v0"]))

    if args.compare:
        other = (args.compare_launcher.split() + [str(args.compare), "-n", args.n,
                                                  "-t", TARGETS, "-s", "7"]
                 + args.compare_args.split())
        for chunk in ("2^16", "2^20", "1000003"):
            ok &= check(f"{args.compare.name} {args.compare_args} == {args.bin.name} "
                        f"(chunk {chunk})",
                        counts(ref + ["-c", chunk]), counts(other + ["-c", chunk]))

    print("all reproducibility checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
