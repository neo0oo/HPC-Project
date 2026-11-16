#!/usr/bin/env python3
"""Correctness check: observed target counts vs theory.

Runs a binary for several seeds and fails (exit 1) if any target's
z = (observed - expected) / sqrt(expected) has |z| > threshold, or if the
binary's own `expected` column disagrees with analysis/theory.py.

Counts are Poisson-like, so sqrt(expected) is the standard deviation and
|z| > 4 is very unlikely (< 1e-4) for a correct generator.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

from theory import expected_count, expected_count_chunked

ROOT = Path(__file__).resolve().parent.parent


def run(cmd_prefix: list[str], binary: Path, n: int, targets: str, seed: int,
        kernel: str, chunk: str | None) -> list[dict]:
    cmd = [*cmd_prefix, str(binary), "-n", str(n), "-t", targets,
           "-s", str(seed), "-k", kernel, "-H"]
    if chunk:
        cmd += ["-c", chunk]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=ROOT / "bin" / "serial")
    ap.add_argument("--launcher", default="",
                    help="command prefix, e.g. 'srun -n 4' or 'mpirun -n 4'")
    ap.add_argument("-n", type=lambda s: int(float(s)), default=200_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--targets", default="a,the,love,romeo")
    ap.add_argument("--kernel", choices=["chunked", "v0", "fused"], default="chunked")
    ap.add_argument("--chunk", default=None, help="passed to -c (e.g. 2^20)")
    ap.add_argument("--threshold", type=float, default=4.0)
    args = ap.parse_args()

    if not args.bin.exists():
        print(f"error: {args.bin} not found (run `make`)", file=sys.stderr)
        return 2

    n_targets = len(args.targets.split(","))
    failures = 0
    print(f"{args.bin.name} kernel={args.kernel} n={args.n:.3g}")
    print(f"{'seed':>4} {'target':>8} {'count':>12} {'expected':>16} {'z':>8}  status")

    for seed in args.seeds:
        rows = run(args.launcher.split(), args.bin, args.n, args.targets, seed,
                   args.kernel, args.chunk)
        if len(rows) != n_targets:
            print(f"FAIL seed {seed}: expected {n_targets} rows, got {len(rows)}")
            failures += 1
            continue
        for r in rows:
            n, length, obs = int(r["n"]), int(r["len"]), int(r["count"])
            if r["kernel"] == "v0":
                exp = expected_count(n, length)
            else:
                exp = expected_count_chunked(n, int(r["chunk"]), length)
            problems = []
            if not math.isclose(exp, float(r["expected"]), rel_tol=1e-6):
                problems.append(f"binary expected={r['expected']}")
            z = (obs - exp) / math.sqrt(exp) if exp > 0 else (math.inf if obs else 0.0)
            if abs(z) > args.threshold:
                problems.append(f"|z|>{args.threshold:g}")
            failures += bool(problems)
            status = "ok" if not problems else "FAIL " + ", ".join(problems)
            print(f"{seed:>4} {r['target']:>8} {obs:>12} {exp:>16.3f} {z:>8.3f}  {status}")

    if failures:
        print(f"\nFAILED: {failures} check(s)")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
