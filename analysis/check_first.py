#!/usr/bin/env python3
"""Correctness of first_match.

1. Exactness: the first occurrence position must be IDENTICAL for every
   configuration (threads, check interval K, sync/async, kernel, ranks),
   because the global text does not depend on how it is split.
2. Theory: over many seeds, the mean waiting time must agree with Conway's
   expected wait (theory.expected_wait). The waiting time is ~geometric, so
   its standard deviation is ~ its mean and the mean of N runs has standard
   error ~ E/sqrt(N); we require |mean - E| < 4 E / sqrt(N).
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import subprocess
import sys
from pathlib import Path

from theory import expected_wait

ROOT = Path(__file__).resolve().parent.parent


def first_pos(cmd: list[str], env: dict) -> int:
    out = subprocess.run(cmd + ["-H"], check=True, capture_output=True,
                         text=True, env=env).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    return int(rows[0]["first_pos"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=ROOT / "bin" / "first_match")
    ap.add_argument("--launcher", default="", help="e.g. 'srun -n 4'")
    ap.add_argument("--target", default="romeo")
    ap.add_argument("--seeds", type=int, default=40)
    args = ap.parse_args()

    launch = args.launcher.split()
    base = launch + [str(args.bin), "-t", args.target, "-n", "1e13", "-c", "2^16"]
    ok = True

    configs = [
        ("1 thread, K=1, sync, chunked", 1, ["-K", "1", "-m", "sync", "-k", "chunked"]),
        ("3 threads, K=4, sync, fused", 3, ["-K", "4", "-m", "sync", "-k", "fused"]),
        ("4 threads, K=1, async, fused", 4, ["-K", "1", "-m", "async", "-k", "fused"]),
        ("2 threads, K=64, async, chunked", 2, ["-K", "64", "-m", "async", "-k", "chunked"]),
    ]
    for seed in (1, 2, 3):
        got = {}
        for name, thr, extra in configs:
            env = dict(os.environ, OMP_NUM_THREADS=str(thr))
            got[name] = first_pos(base + ["-s", str(seed)] + extra, env)
        same = len(set(got.values())) == 1 and 0 not in got.values()
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} seed {seed}: first_pos identical across "
              f"configs = {sorted(set(got.values()))}")

    env = dict(os.environ, OMP_NUM_THREADS="2")
    waits = [first_pos(base + ["-s", str(s), "-k", "fused", "-K", "4"], env)
             for s in range(100, 100 + args.seeds)]
    E = expected_wait(args.target)
    mean = sum(waits) / len(waits)
    tol = 4 * E / math.sqrt(len(waits))
    good = abs(mean - E) < tol
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} mean wait over {len(waits)} seeds = {mean:.4g}, "
          f"Conway E = {E:.4g} (tolerance {tol:.2g})")

    print("all first_match checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
