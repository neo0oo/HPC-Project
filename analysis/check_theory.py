#!/usr/bin/env python3
"""Correctness check: observed target counts vs (n - L + 1) / 27^L.

Runs ./serial for several seeds and fails (exit 1) if any target's
z = (observed - expected) / sqrt(expected) has |z| > threshold.
"""
import argparse
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = 27


def expected_count(n: int, length: int) -> float:
    return max(n - length + 1, 0) / K**length


def run(binary: Path, n: int, targets: str, seed: int) -> list[dict]:
    cmd = [str(binary), "-n", str(n), "-t", targets, "-s", str(seed), "-H"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=ROOT / "serial")
    ap.add_argument("-n", type=lambda s: int(float(s)), default=200_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--targets", default="a,the,love,romeo")
    ap.add_argument("--threshold", type=float, default=4.0)
    args = ap.parse_args()

    if not args.bin.exists():
        print(f"error: {args.bin} not found (run `make serial`)", file=sys.stderr)
        return 2

    n_targets = len(args.targets.split(","))
    failures = 0
    print(f"{'seed':>4} {'target':>8} {'count':>12} {'expected':>16} {'z':>8}  status")

    for seed in args.seeds:
        rows = run(args.bin, args.n, args.targets, seed)
        if len(rows) != n_targets:
            print(f"FAIL seed {seed}: expected {n_targets} rows, got {len(rows)}")
            failures += 1
            continue
        for r in rows:
            n, length, obs = int(r["n"]), int(r["len"]), int(r["count"])
            exp = expected_count(n, length)
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
