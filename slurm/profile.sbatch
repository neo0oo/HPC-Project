#!/bin/bash
# Phase 5: profiling / bottleneck evidence, single core.
#
#     sbatch slurm/profile.sbatch
#
# bin/prof times each kernel and a generate-only pass ("gen"), so
#     matching cost = kernel - gen        (application instrumentation)
# bin/prof_papi adds hardware counters (instructions, cycles, IPC, branch
# mispredictions, L1D misses per character) when the PAPI module and the
# kernel's perf_event settings allow it; otherwise those columns stay empty.
#
# Output: results/raw/profile/<id>.csv  (prof) and <id>_papi.csv (PAPI)
#
#SBATCH --job-name=profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N="${N:-2e9}"
REPS="${REPS:-3}"
five() { targets five_letter | head -"$1" | paste -sd, -; }

profile_all() { # profile_all <binary> <csv>
    for rep in $(seq 1 "$REPS"); do
        append_csv "$2" "$1" -n "$N" -k gen -t romeo -s "$rep"
        for k in chunked fused; do
            for nt in 1 4 16; do
                append_csv "$2" "$1" -n "$N" -k "$k" -t "$(five "$nt")" -s "$rep"
            done
        done
    done
}

out=$(csv_for profile)
echo "== timing breakdown (prof)"
profile_all "$BIN/prof" "$out"

echo "== hardware counters (prof_papi)"
echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
(
    module load PAPI/7.1.0-GCCcore-13.2.0 &&
    make -s BIN="$BIN" CC=gcc papi &&
    papi_avail -a 2>/dev/null | head -40 &&
    profile_all "$BIN/prof_papi" "${out%.csv}_papi.csv"
) || echo "PAPI run failed; timing breakdown above is the profiling evidence"
echo "done: job $SLURM_JOB_ID"
