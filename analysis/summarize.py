#!/usr/bin/env python3
"""Summarize every experiment found under results/raw (stdlib only).

    python3 analysis/summarize.py > results/summary.txt
    python3 analysis/summarize.py --plots results/figures   # needs matplotlib
        (on the cluster: module load matplotlib/3.7.2-gfbf-2023a)

Each section reads all CSVs of one experiment (all job ids) and prints medians
over repetitions. Sections with no data are skipped, so it can be run at any
point while jobs are still finishing.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"  # --raw overrides
sys.path.insert(0, str(Path(__file__).resolve().parent))
from theory import expected_wait  # noqa: E402

PLOTS: dict[str, callable] = {}


def rows_of(exp: str, pattern: str = "*.csv") -> list[dict]:
    out = []
    for f in sorted((RAW / exp).glob(pattern)):
        if f.name.endswith("_cpu.csv") or f.name.endswith("_papi.csv"):
            continue
        out += list(csv.DictReader(open(f)))
    return out


def runs_of(rows: list[dict]) -> list[dict]:
    """Count CSVs have one row per target per run; keep one row per run and
    attach the number of targets."""
    seen, out = {}, []
    for r in rows:
        key = tuple(r.get(k) for k in ("version", "kernel", "ranks", "threads", "n",
                                       "chunk", "seed", "time_s", "host", "layout",
                                       "schedule", "flags"))
        if key in seen:
            seen[key]["ntargets"] += 1
        else:
            r = dict(r, ntargets=1)
            seen[key] = r
            out.append(r)
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def imbalance(r) -> float | None:
    """busy_max / busy_min; inf when some worker was (almost) idle, i.e. there
    were fewer chunks than workers (printed as 'idle')."""
    lo, hi = f(r, "busy_min_s"), f(r, "busy_max_s")
    if lo is None or hi is None or hi <= 0:
        return None
    return hi / lo if lo > 0.01 * hi else math.inf


def fmt_imb(x: float) -> str:
    return "  idle" if x == math.inf else f"{x:>6.3f}" if x == x else "     -"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------- scaling ---
def scaling(exp: str, mode: str, label: str, sub: str = "*.csv") -> list[dict]:
    runs = runs_of(rows_of(exp, sub))
    if not runs:
        return []
    by = defaultdict(list)
    for r in runs:
        by[int(r["ranks"]) * int(r["threads"])].append(r)
    if 1 not in by:
        print(f"{label}: no 1-worker run yet")
        return []
    t1 = med(f(r, "time_s") for r in by[1])
    print(f"\n{label}   (T1 = {t1:.2f} s, median of {len(by[1])})")
    print(f"{'workers':>8} {'time_s':>9} {'speedup':>8} {'eff':>6} {'imbal':>6} "
          f"{'comm_s':>9} {'chars/s':>10} reps")
    table = []
    for w in sorted(by):
        rs = by[w]
        tp = med(f(r, "time_s") for r in rs)
        e = (t1 / tp / w) if mode == "strong" else t1 / tp
        s = e * w
        imb = med(imbalance(r) for r in rs)
        comm = med(f(r, "comm_s") for r in rs)
        rate = med(f(r, "chars_per_s") for r in rs)
        table.append(dict(w=w, time=tp, speedup=s, eff=e, imb=imb, comm=comm, rate=rate))
        print(f"{w:>8} {tp:>9.3f} {s:>8.2f} {e:>6.3f} {fmt_imb(imb)} "
              f"{comm if comm == comm else 0:>9.2e} {rate:>10.3e} {len(rs)}")
    return table


def sec_scaling() -> None:
    section("STRONG / WEAK SCALING (speedup vs the 1-worker run of the same job)")
    tabs = {}
    tabs["omp strong"] = scaling("omp_strong", "strong", "OpenMP strong (threads)")
    tabs["mpi strong"] = scaling("mpi_strong", "strong", "MPI strong (ranks, 1 node)")
    weak = {}
    weak["omp close"] = scaling("omp_weak/close", "weak", "OpenMP weak, close binding")
    weak["omp spread"] = scaling("omp_weak/spread", "weak", "OpenMP weak, spread binding")
    weak["mpi"] = scaling("mpi_weak", "weak", "MPI weak (1 node)")

    def plot(plt, out):
        for name, tabset, fname, ylab in (("strong", tabs, "scaling_strong.png", "speedup"),
                                          ("weak", weak, "scaling_weak.png", "efficiency")):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for lab, tab in tabset.items():
                if not tab:
                    continue
                w = [t["w"] for t in tab]
                ax[0].plot(w, [t["speedup"] for t in tab], "o-", label=lab)
                ax[1].plot(w, [t["eff"] for t in tab], "o-", label=lab)
            allw = sorted({t["w"] for tab in tabset.values() for t in tab})
            if allw:
                ax[0].plot(allw, allw, "k:", label="ideal")
            for a in ax:
                a.set_xscale("log", base=2)
                a.set_xlabel("cores (ranks x threads)")
                a.grid(alpha=.3)
                a.legend(fontsize=8)
            ax[0].set_yscale("log", base=2)
            ax[0].set_ylabel("speedup" if name == "strong" else "scaled speedup")
            ax[1].set_ylabel("parallel efficiency")
            ax[1].set_ylim(0, 1.1)
            fig.suptitle(f"{name} scaling")
            fig.tight_layout()
            fig.savefig(out / fname, dpi=150)
    PLOTS["scaling"] = plot


def sec_layouts() -> None:
    runs = runs_of(rows_of("mpi_2node"))
    if runs:
        section("ONE NODE vs TWO NODES (MPI, same work)")
        by = defaultdict(list)
        for r in runs:
            by[r["layout"]].append(r)
        print(f"{'layout':>14} {'time_s':>9} {'comm_s':>10} {'chars/s':>10} reps")
        for k, rs in by.items():
            print(f"{k:>14} {med(f(r,'time_s') for r in rs):>9.3f} "
                  f"{med(f(r,'comm_s') for r in rs):>10.2e} "
                  f"{med(f(r,'chars_per_s') for r in rs):>10.3e} {len(rs)}")
    runs = runs_of(rows_of("hybrid"))
    if runs:
        section("HYBRID: ranks x threads on 2 nodes x 16 cores")
        by = defaultdict(list)
        for r in runs:
            by[(int(r["ranks"]), int(r["threads"]))].append(r)
        print(f"{'ranks x thr':>12} {'time_s':>9} {'imbal':>6} {'comm_s':>10} "
              f"{'chars/s':>10} reps")
        for (rk, th), rs in sorted(by.items(), reverse=True):
            imb = med(imbalance(r) for r in rs)
            print(f"{f'{rk} x {th}':>12} {med(f(r,'time_s') for r in rs):>9.3f} "
                  f"{fmt_imb(imb)} {med(f(r,'comm_s') for r in rs):>10.2e} "
                  f"{med(f(r,'chars_per_s') for r in rs):>10.3e} {len(rs)}")


# ---------------------------------------------------------- first match ---
def sec_first_match() -> None:
    rows = rows_of("first_match")
    if not rows:
        return
    section("TIME TO SOLUTION (first_match)")
    full = [r for r in rows if int(r["ranks"]) * int(r["threads"]) >= 32]
    print("A. waiting time vs Conway's expectation (32 cores)")
    print(f"{'target':>10} {'L':>2} {'runs':>4} {'mean wait':>11} {'Conway E':>11} "
          f"{'mean/E':>7} {'sd/mean':>7} {'median t_s':>10}")
    ecdf = {}
    for t in ("romeo", "juliet", "capulet", "montague", "wherefore"):
        rs = [r for r in full if r["target"] == t and int(r["first_pos"]) > 0]
        if not rs:
            continue
        w = [int(r["first_pos"]) for r in rs]
        E = expected_wait(t)
        m = st.mean(w)
        sd = st.stdev(w) if len(w) > 1 else float("nan")
        ecdf[t] = (sorted(w), E)
        print(f"{t:>10} {len(t):>2} {len(w):>4} {m:>11.4g} {E:>11.4g} {m/E:>7.3f} "
              f"{sd/m:>7.3f} {med(f(r,'time_s') for r in rs):>10.3f}")
    print("   (waiting time ~ geometric: expect mean/E ~ 1 within ~2/sqrt(runs), sd/mean ~ 1)")

    cap = [r for r in rows if r["target"] == "capulet" and int(r["seed"]) > 100]
    if cap:
        print("\nB. capulet time-to-solution vs cores (same 20 seeds per point)")
        by = defaultdict(list)
        for r in cap:
            by[int(r["ranks"]) * int(r["threads"])].append(r)
        # Waiting times vary ~geometrically between seeds, so pair runs by
        # seed: speedup = median over seeds of t1(seed)/tp(seed), and also the
        # total-time ratio (sum t1 / sum tp over the common seeds).
        one = {r["seed"]: f(r, "time_s") for r in by.get(1, [])}
        print(f"{'cores':>6} {'median t_s':>10} {'speedup':>8} {'eff':>6} "
              f"{'sum-speedup':>11} {'wasted':>7}")
        for c in sorted(by):
            tp = {r["seed"]: f(r, "time_s") for r in by[c]}
            common = [k for k in tp if k in one]
            if not common:
                continue
            sp = med(one[k] / tp[k] for k in common)
            tot = sum(one[k] for k in common) / sum(tp[k] for k in common)
            print(f"{c:>6} {med(tp.values()):>10.3f} {sp:>8.2f} {sp/c:>6.3f} "
                  f"{tot:>11.2f} {med(f(r,'wasted_frac') for r in by[c]):>7.4f}")
    wf = [r for r in rows if r["target"] == "wherefore"]
    if wf:
        print("\nC. 'wherefore' on 32 cores")
        for r in wf:
            print(f"   seed {r['seed']}: found after {int(r['first_pos']):.3e} chars "
                  f"in {float(r['time_s']):.1f} s ({float(r['chars_per_s']):.3e} chars/s)")

    def plot(plt, out):
        if not ecdf:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for t, (w, E) in ecdf.items():
            x = [v / E for v in w]
            ax.step(x, [(i + 1) / len(x) for i in range(len(x))], where="post", label=t)
        xs = [i / 100 for i in range(0, 501)]
        ax.plot(xs, [1 - math.exp(-x) for x in xs], "k--", label="geometric (theory)")
        ax.set_xlabel("first occurrence / Conway expected wait")
        ax.set_ylabel("fraction of runs")
        ax.set_xlim(0, 5)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "first_match_ecdf.png", dpi=150)
    PLOTS["first_match"] = plot

    rows = rows_of("first_match_k")
    if rows:
        print("\nD. stop-check interval K, sync vs async (montague, 2 x 16 cores)")
        by = defaultdict(list)
        for r in rows:
            by[(int(r["check"]), r["mode"])].append(r)
        print(f"{'K':>5} {'mode':>6} {'median t_s':>10} {'comm_s':>9} {'wasted':>8} "
              f"{'rounds':>8} {'chars/s':>10}")
        for (K, m), rs in sorted(by.items()):
            print(f"{K:>5} {m:>6} {med(f(r,'time_s') for r in rs):>10.3f} "
                  f"{med(f(r,'comm_s') for r in rs):>9.4f} "
                  f"{med(f(r,'wasted_frac') for r in rs):>8.4f} "
                  f"{med(f(r,'rounds') for r in rs):>8.0f} "
                  f"{med(f(r,'chars_per_s') for r in rs):>10.3e}")


# --------------------------------------------------------- optimization ---
def sec_optimize() -> None:
    runs = runs_of(rows_of("optimize_1core"))
    if runs:
        section("OPTIMIZATION: chunked (baseline) vs fused kernel")
        by = defaultdict(list)
        for r in runs:  # first target of the mixed-length set is "a" (L=1)
            label = str(r["ntargets"]) if r["len"] == "5" else f"mixed{r['ntargets']}"
            by[(r["kernel"], label)].append(r)
        print("A. single core")
        print(f"{'targets':>8} {'chunked chars/s':>16} {'fused chars/s':>14} {'speedup':>8}")
        for lab in sorted({l for _, l in by}, key=lambda x: (x.startswith("mixed"), int(x.strip("mixed")))):
            c = med(f(r, "chars_per_s") for r in by.get(("chunked", lab), []))
            fu = med(f(r, "chars_per_s") for r in by.get(("fused", lab), []))
            print(f"{lab:>8} {c:>16.3e} {fu:>14.3e} {fu/c:>8.2f}")
    runs = runs_of(rows_of("optimize_32"))
    if runs:
        print("\nB. whole allocation (OpenMP)")
        by = defaultdict(list)
        for r in runs:
            by[(r["kernel"], r["ntargets"], int(r["threads"]))].append(r)
        for (k, nt, th), rs in sorted(by.items()):
            print(f"   {k:>8} {nt:>2} targets {th:>3} threads: "
                  f"{med(f(r,'chars_per_s') for r in rs):.3e} chars/s")
    runs = runs_of(rows_of("optimize_flags"))
    if runs:
        print("\nC. compiler flags (single core, 1 target)")
        by = defaultdict(list)
        for r in runs:
            by[(r["flags"], r["kernel"])].append(r)
        for (fl, k), rs in sorted(by.items()):
            print(f"   {fl:>32} {k:>8}: {med(f(r,'chars_per_s') for r in rs):.3e} chars/s")


def sec_sensitivity() -> None:
    runs = runs_of(rows_of("sensitivity"))
    if not runs:
        return
    section("SENSITIVITY: chunk size x OpenMP schedule (fused, all cores)")
    by = defaultdict(list)
    for r in runs:
        by[(int(r["chunk"]), r["schedule"])].append(r)
    scheds = sorted({s for _, s in by})
    print(f"{'chunk':>8} " + " ".join(f"{s+' t_s':>14} {'imbal':>6}" for s in scheds))
    for c in sorted({c for c, _ in by}):
        cells = []
        for s in scheds:
            rs = by.get((c, s), [])
            imb = med(imbalance(r) for r in rs)
            cells.append(f"{med(f(r,'time_s') for r in rs):>14.3f} {fmt_imb(imb)}")
        print(f"{'2^'+str(int(math.log2(c))):>8} " + " ".join(cells))


def sec_profile() -> None:
    for name, pat in (("timing", "*.csv"), ("PAPI", "*_papi.csv")):
        rows = []
        for fpath in sorted((RAW / "profile").glob(pat)):
            if name == "timing" and fpath.name.endswith("_papi.csv"):
                continue
            rows += list(csv.DictReader(open(fpath)))
        if not rows:
            continue
        if name == "timing":
            section("PROFILE (single core)")
        by = defaultdict(list)
        for r in rows:
            by[(r["kernel"], int(r["ntargets"]))].append(r)
        gen = med(f(r, "ns_per_char") for r in by.get(("gen", 0), []))
        print(f"\n{name}: generation only = {gen:.3f} ns/char")
        print(f"{'kernel':>8} {'nt':>3} {'ns/char':>8} {'match ns':>9} {'ins/char':>9} "
              f"{'cyc/char':>9} {'IPC':>6} {'brmsp/kc':>9} {'L1miss/kc':>9}")
        for (k, nt), rs in sorted(by.items()):
            ns = med(f(r, "ns_per_char") for r in rs)
            cols = [med(f(r, c) for r in rs) for c in
                    ("ins_per_char", "cyc_per_char", "ipc", "br_msp_per_kchar", "l1_dcm_per_kchar")]
            cs = " ".join(f"{c:>9.3f}" if c == c else f"{'-':>9}" for c in cols)
            print(f"{k:>8} {nt:>3} {ns:>8.3f} {ns-gen if k!='gen' else 0:>9.3f} {cs}")


def sec_gpu() -> None:
    found = False
    for exp, key, label in (("gpu_block", None, "threads per block"),
                            ("gpu_chunk", "chunk", "chunk"),
                            ("gpu_targets", "ntargets", "targets"),
                            ("gpu_headline", None, "headline"),
                            ("gpu_openacc", "version", "OpenACC vs OpenMP")):
        runs = runs_of(rows_of(exp))
        if not runs:
            continue
        if not found:
            section("GPU (CUDA) and OpenACC")
            found = True
        print(f"\n{exp}")
        by = defaultdict(list)
        for r in runs:
            if exp == "gpu_block":
                k = (r["host"], r["threads"])
            elif exp == "gpu_headline":
                k = (r["host"], r["n"])
            else:
                k = (r["host"], r[key])
            by[k].append(r)
        for k, rs in sorted(by.items(), key=lambda kv: (kv[0][0], float(kv[0][1]) if kv[0][1].replace('.', '').isdigit() else 0)):
            print(f"   {k[0]:>8} {label} {k[1]:>14}: {med(f(r,'chars_per_s') for r in rs):.3e} chars/s "
                  f"({len(rs)} runs, t={med(f(r,'time_s') for r in rs):.3f} s)")
        if exp == "gpu_headline":
            for r in rows_of(exp):
                print(f"   {r['host']}: '{r['target']}' x{r['count']} in {float(r['n']):.3e} chars "
                      f"(expected {float(r['expected']):.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", type=Path, default=None)
    ap.add_argument("--raw", type=Path, default=None, help="default results/raw")
    args = ap.parse_args()
    global RAW
    if args.raw:
        RAW = args.raw
    for fn in (sec_scaling, sec_layouts, sec_first_match, sec_optimize,
               sec_sensitivity, sec_profile, sec_gpu):
        try:
            fn()
        except Exception as e:  # keep going: partial data is normal
            print(f"\n[{fn.__name__} skipped: {type(e).__name__}: {e}]")
    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available: no plots", file=sys.stderr)
            return 0
        args.plots.mkdir(parents=True, exist_ok=True)
        for name, p in PLOTS.items():
            p(plt, args.plots)
        print(f"plots written to {args.plots}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
