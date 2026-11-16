#!/usr/bin/env python3
"""Sample real CPU frequency and utilization while a benchmark runs (Linux).

Started in the background by slurm/common.sh (run_sampled) and stopped with
SIGTERM when the benchmark ends. It watches only the CPUs this job may use
(its affinity mask), so other users on a shared node are ignored.

- utilization per CPU from /proc/stat deltas (start -> stop)
- frequency per CPU from /proc/cpuinfo "cpu MHz" (APERF/MPERF based, i.e.
  the actual clock), sampled every --interval seconds

"Busy" CPUs are those with utilization > 50%. Appends one CSV row:
    label,busy_cpus,mean_mhz_busy,min_mhz_busy,max_mhz_busy,mean_util_busy,samples
This is the evidence for frequency effects in the scaling results (all-core
boost is lower than single-core boost).
"""
from __future__ import annotations

import argparse
import os
import signal
import time

HEADER = "label,busy_cpus,mean_mhz_busy,min_mhz_busy,max_mhz_busy,mean_util_busy,samples"


def read_stat(cpus: set[int]) -> dict[int, tuple[int, int]]:
    """cpu -> (busy_ticks, total_ticks)."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            cpu = int(parts[0][3:])
            if cpu not in cpus:
                continue
            v = list(map(int, parts[1:]))
            idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
            out[cpu] = (sum(v) - idle, sum(v))
    return out


def read_mhz(cpus: set[int]) -> dict[int, float]:
    out, cpu = {}, None
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("processor"):
                cpu = int(line.split(":")[1])
            elif line.startswith("cpu MHz") and cpu in cpus:
                out[cpu] = float(line.split(":")[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    linux = os.path.exists("/proc/stat") and hasattr(os, "sched_getaffinity")
    cpus = set(os.sched_getaffinity(0)) if linux else set()
    stop = False

    def on_term(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)

    start = read_stat(cpus) if linux else {}
    sums = {c: 0.0 for c in cpus}
    samples = 0
    time.sleep(0.2)  # first sample early so short runs get at least one
    while not stop:
        if linux:
            for c, mhz in read_mhz(cpus).items():
                sums[c] += mhz
            samples += 1
        t_end = time.time() + args.interval
        while not stop and time.time() < t_end:
            time.sleep(0.05)
    end = read_stat(cpus) if linux else {}

    util = {}
    for c in cpus:
        if c in start and c in end:
            db, dt = end[c][0] - start[c][0], end[c][1] - start[c][1]
            util[c] = db / dt if dt > 0 else 0.0
    busy = [c for c, u in util.items() if u > 0.5]
    mhz = [sums[c] / samples for c in busy] if samples else []

    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a") as f:
        if new:
            f.write(HEADER + "\n")
        if mhz:
            f.write(f"{args.label},{len(busy)},{sum(mhz)/len(mhz):.0f},{min(mhz):.0f},"
                    f"{max(mhz):.0f},{sum(util[c] for c in busy)/len(busy):.3f},{samples}\n")
        else:
            f.write(f"{args.label},{len(busy)},,,,,{samples}\n")


if __name__ == "__main__":
    main()
