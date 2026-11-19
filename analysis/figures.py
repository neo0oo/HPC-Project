#!/usr/bin/env python3
"""Report figures not drawn by summarize.py (needs matplotlib).

    python3 analysis/figures.py                 # writes results/figures/*.png

  optimization_targets.png  chars/s vs number of targets: baseline, fused v1, fused v2
  stop_check_k.png          first_match: time and comm vs stop-check interval K
  time_to_solution.png      capulet: time-to-solution vs cores (per-seed speedup)
  sensitivity_chunk.png     chunk size x OpenMP schedule: time and imbalance
  throughput_platforms.png  best throughput per platform (1 core .. GPU)
  gpu_sweeps.png            GPU block size / chunk size / targets, both cards
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OLD = ROOT / "results" / "old" / "v1"
OUT = ROOT / "results" / "figures"
GPU_NAME = {"fortran": "RTX PRO 6000 Blackwell", "haskell": "RTX 6000 Ada"}


def rows(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.csv")):
        if not f.name.endswith(("_cpu.csv", "_papi.csv")):
            out += list(csv.DictReader(open(f)))
    return out


def per_run(rs: list[dict]) -> list[dict]:
    """One row per run (count CSVs have one row per target); adds ntargets."""
    seen, out = {}, []
    for r in rs:
        key = tuple(r.get(k) for k in ("version", "kernel", "ranks", "threads", "n", "chunk",
                                       "seed", "time_s", "host", "schedule", "flags", "block"))
        if key in seen:
            seen[key]["nt"] += 1
        else:
            seen[key] = dict(r, nt=1)
            out.append(seen[key])
    return out


def med(xs):
    xs = list(xs)
    return st.median(xs) if xs else float("nan")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_optimization():
    def series(d, kernel):
        by = defaultdict(list)
        for r in per_run(rows(d)):
            if r["kernel"] == kernel and r["nt"] in (1, 2, 4, 8, 16) and r["len"] == "5":
                # (the mixed-length 8-target set starts with a 1-letter target)
                by[r["nt"]].append(float(r["chars_per_s"]))
        return sorted(by), [med(by[k]) for k in sorted(by)]

    fig, ax = plt.subplots(figsize=(6, 4))
    for d, k, lab, st_ in ((RAW / "optimize_1core", "chunked", "baseline (chunked)", "o-"),
                           (OLD / "optimize_1core", "fused", "fused v1 (grouped matcher)", "s--"),
                           (RAW / "optimize_1core", "fused", "fused v2 (+ plain loop for <=3)", "^-")):
        if d.exists():
            x, y = series(d, k)
            ax.plot(x, y, st_, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of 5-letter targets")
    ax.set_ylabel("characters / s (1 core)")
    ax.set_title("Optimization: one fused pass over all targets")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "optimization_targets.png")


def fig_stop_check():
    rs = rows(RAW / "first_match_k")
    if not rs:
        return
    by = defaultdict(list)
    for r in rs:
        by[(r["mode"], int(r["check"]))].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mode, mk in (("sync", "o-"), ("async", "s-")):
        ks = sorted(k for m, k in by if m == mode)
        ax[0].plot(ks, [med(float(r["time_s"]) for r in by[(mode, k)]) for k in ks], mk, label=mode)
        ax[1].plot(ks, [med(float(r["comm_s"]) for r in by[(mode, k)]) for k in ks], mk,
                   label=f"{mode}: comm time")
        ax[1].plot(ks, [med(float(r["wasted_frac"]) for r in by[(mode, k)]) for k in ks],
                   mk.replace("-", ":"), label=f"{mode}: wasted fraction")
    ax[0].set_ylabel("median time to first 'montague' (s)")
    ax[1].set_ylabel("seconds / fraction")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("stop-check interval K (chunks per thread)")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle("Stop check: communication vs wasted work (2 nodes x 16 cores)")
    save(fig, "stop_check_k.png")


def fig_time_to_solution():
    rs = [r for r in rows(RAW / "first_match") if r["target"] == "capulet" and int(r["seed"]) > 100]
    if not rs:
        return
    t = defaultdict(dict)
    for r in rs:
        t[int(r["ranks"]) * int(r["threads"])][r["seed"]] = float(r["time_s"])
    cores = sorted(t)
    sp = [med(t[1][s] / t[c][s] for s in t[c] if s in t[1]) for c in cores]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cores, sp, "o-", label="median over 20 seeds of t1/tp")
    ax.plot(cores, cores, "k:", label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("cores")
    ax.set_ylabel("speedup in time to first 'capulet'")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    save(fig, "time_to_solution.png")


def fig_sensitivity():
    rs = rows(RAW / "sensitivity")
    if not rs:
        return
    by = defaultdict(list)
    for r in rs:
        by[(r["schedule"], int(r["chunk"]))].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for sched in sorted({s for s, _ in by}):
        cs = sorted(c for s, c in by if s == sched)
        ax[0].plot(cs, [med(float(r["time_s"]) for r in by[(sched, c)]) for c in cs], "o-", label=sched)
        imb = []
        for c in cs:
            vals = []
            for r in by[(sched, c)]:
                lo, hi = float(r["busy_min_s"]), float(r["busy_max_s"])
                vals.append(hi / lo if lo > 0 else float("nan"))
            imb.append(med(vals))
        ax[1].plot(cs, imb, "o-", label=sched)
    ax[0].set_ylabel("time (s)")
    ax[1].set_ylabel("load imbalance (max/min busy time)")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("chunk size (characters)")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle("Sensitivity: chunk size x OpenMP schedule (fused, 32 threads)")
    save(fig, "sensitivity_chunk.png")


def best(d: Path, pred) -> float:
    vals = [float(r["chars_per_s"]) for r in per_run(rows(d)) if pred(r)]
    return med(vals) if vals else float("nan")


def fig_platforms():
    one = lambda r: r["nt"] == 1  # noqa: E731
    bars = [
        ("1 core\nbaseline kernel", best(RAW / "optimize_1core", lambda r: one(r) and r["kernel"] == "chunked")),
        ("1 core\nfused", best(RAW / "optimize_1core", lambda r: one(r) and r["kernel"] == "fused")),
        ("32 cores, OpenMP\n(fused)", best(RAW / "optimize_32", lambda r: one(r) and r["kernel"] == "fused")),
        ("2 nodes, MPI 32\n(baseline kernel)", best(RAW / "mpi_2node", lambda r: r.get("layout") == "16+16_2nodes")),
        ("GPU Ada\nOpenACC", best(RAW / "gpu_openacc", lambda r: r["version"] == "acc_gpu" and r["host"] == "haskell")),
        ("GPU Ada\nCUDA", best(RAW / "gpu_targets", lambda r: one(r) and r["host"] == "haskell")),
        ("GPU Blackwell\nOpenACC", best(RAW / "gpu_openacc", lambda r: r["version"] == "acc_gpu" and r["host"] == "fortran")),
        ("GPU Blackwell\nCUDA", best(RAW / "gpu_targets", lambda r: one(r) and r["host"] == "fortran")),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(bars))
    ax.bar(x, [b[1] for b in bars], color="#4a7bb7")
    for i, (_, v) in enumerate(bars):
        ax.text(i, v * 1.15, f"{v:.2g}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("characters / s (1 target)")
    ax.set_title("Throughput per platform")
    ax.grid(alpha=.3, axis="y")
    save(fig, "throughput_platforms.png")


def fig_gpu():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for host in ("fortran", "haskell"):
        lab = GPU_NAME[host]
        for a, d, key, xl in ((ax[0], "gpu_block", "block", "threads per block"),
                              (ax[1], "gpu_chunk", "chunk", "chunk size"),
                              (ax[2], "gpu_targets", "nt", "number of targets")):
            by = defaultdict(list)
            for r in per_run(rows(RAW / d)):
                if r["host"] == host and r.get(key):
                    by[int(r[key])].append(float(r["chars_per_s"]))
            ks = sorted(by)
            a.plot(ks, [med(by[k]) for k in ks], "o-", label=lab)
            a.set_xscale("log", base=2)
            a.set_xlabel(xl)
            a.grid(alpha=.3)
            a.legend(fontsize=8)
    ax[0].set_ylabel("characters / s")
    fig.suptitle("CUDA kernel sweeps")
    save(fig, "gpu_sweeps.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_optimization, fig_stop_check, fig_time_to_solution, fig_sensitivity,
              fig_platforms, fig_gpu):
        try:
            f()
        except Exception as e:  # partial data is fine
            print(f"{f.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
