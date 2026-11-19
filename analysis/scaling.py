#!/usr/bin/env python3
"""Summarize OpenMP strong / weak scaling runs (stdlib only; plots optional).

    python3 analysis/scaling.py strong results/raw/omp_strong/<id>.csv
    python3 analysis/scaling.py weak   results/raw/omp_weak/close/<id>.csv \
                                       results/raw/omp_weak/spread/<id>.csv

For each thread count: median time over repetitions, speedup S = T1/Tp
(strong) or efficiency E = T1/Tp (weak, work per thread fixed), parallel
efficiency, load imbalance busy_max/busy_min, and, when the matching
<id>_cpu.csv exists, the measured clock of the busy cores.

"Frequency-corrected efficiency" = efficiency * (MHz at 1 thread / MHz at p):
how efficient the run would be if the clock had not dropped. If it is ~1,
the loss is explained by turbo/frequency, not by the parallelization.

--plot FILE.png writes a figure if matplotlib is available.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_runs(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    # one row per target per run; keep the first target of each run
    runs, seen = [], set()
    for r in rows:
        key = (r["threads"], r["seed"], r["time_s"])
        if key not in seen:
            seen.add(key)
            runs.append(r)
    return runs


def load_cpu(path: Path) -> dict[tuple[int, int], dict]:
    """(rep, threads) -> sampler row, for rows with places=cores or any bind."""
    cpu = path.with_name(path.stem + "_cpu.csv")
    out = {}
    if not cpu.exists():
        return out
    for r in csv.DictReader(open(cpu)):
        kv = dict(p.split("=") for p in r["label"].split(";"))
        if kv.get("places", "cores") != "cores":
            continue
        out[(int(kv["rep"]), int(kv["threads"]))] = r
    return out


def summarize(path: Path, mode: str) -> list[dict]:
    runs = load_runs(path)
    cpu = load_cpu(path)
    by_t: dict[int, list[dict]] = defaultdict(list)
    for r in runs:
        by_t[int(r["threads"])].append(r)

    def mhz_for(t: int) -> float | None:
        vals = [float(c["mean_mhz_busy"]) for (rep, tt), c in cpu.items()
                if tt == t and c["mean_mhz_busy"]]
        return statistics.median(vals) if vals else None

    if 1 not in by_t:
        sys.exit(f"{path}: no threads=1 run yet ({len(runs)} runs so far) - "
                 "job still running or failed; check results/logs/")
    t1 = statistics.median(float(r["time_s"]) for r in by_t[1])
    mhz1 = mhz_for(1)
    table = []
    for t in sorted(by_t):
        times = [float(r["time_s"]) for r in by_t[t]]
        tp = statistics.median(times)
        imb = statistics.median(float(r["busy_max_s"]) / float(r["busy_min_s"])
                                for r in by_t[t] if float(r["busy_min_s"]) > 0)
        if mode == "strong":
            speedup = t1 / tp
            eff = speedup / t
        else:
            eff = t1 / tp
            speedup = eff * t
        mhz = mhz_for(t)
        corr = eff * (mhz1 / mhz) if mhz and mhz1 else None
        table.append(dict(threads=t, time=tp, tmin=min(times), tmax=max(times),
                          reps=len(times), speedup=speedup, eff=eff, imbalance=imb,
                          mhz=mhz, eff_freq=corr,
                          rate=float(by_t[t][0]["n"]) / tp))
    return table


def print_table(name: str, table: list[dict]) -> None:
    print(f"\n== {name}")
    print(f"{'thr':>5} {'time_s':>9} {'(min-max)':>17} {'speedup':>8} {'eff':>6} "
          f"{'imbal':>6} {'MHz':>6} {'eff@f':>6} {'chars/s':>10}")
    for r in table:
        mhz = f"{r['mhz']:.0f}" if r["mhz"] else "-"
        ef = f"{r['eff_freq']:.3f}" if r["eff_freq"] else "-"
        print(f"{r['threads']:>5} {r['time']:>9.3f} {r['tmin']:>8.3f}-{r['tmax']:<8.3f} "
              f"{r['speedup']:>8.2f} {r['eff']:>6.3f} {r['imbalance']:>6.3f} "
              f"{mhz:>6} {ef:>6} {r['rate']:>10.3e}")


def plot(tables: dict[str, list[dict]], mode: str, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", file=sys.stderr)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, tab in tables.items():
        th = [r["threads"] for r in tab]
        axes[0].plot(th, [r["speedup"] for r in tab], "o-", label=name)
        axes[1].plot(th, [r["eff"] for r in tab], "o-", label=f"{name} measured")
        if any(r["eff_freq"] for r in tab):
            axes[1].plot(th, [r["eff_freq"] or float("nan") for r in tab], "x--",
                         label=f"{name} freq-corrected")
    th = sorted({r["threads"] for tab in tables.values() for r in tab})
    axes[0].plot(th, th, "k:", label="ideal")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("OpenMP threads (1 per core)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("speedup" if mode == "strong" else "scaled speedup")
    axes[1].set_ylabel("parallel efficiency")
    axes[1].set_ylim(0, 1.1)
    fig.suptitle(f"OpenMP {mode} scaling")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["strong", "weak"])
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    tables = {}
    for path in args.csv:
        name = path.parent.name if path.parent.name in ("close", "spread") else path.stem
        tables[name] = summarize(path, args.mode)
        print_table(f"{args.mode}: {path}", tables[name])
    if args.plot:
        plot(tables, args.mode, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
