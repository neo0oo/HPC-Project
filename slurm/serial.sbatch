#!/bin/bash
# Phase 1 baseline: single core, unoptimized kernel.
#
#     sbatch slurm/serial.sbatch
#     REPS=3 N_LEN=2e9 sbatch slurm/serial.sbatch      # quicker variant
#
# Experiments (each repeated REPS times with seeds 1..REPS):
#   baseline_length   one target per length L=1..7, n=N_LEN
#                     -> counts vs theory; does time depend on L? (should not)
#   baseline_ntargets 1,2,4,8,16 five-letter targets, n=N_TC
#                     -> cost per extra target (one pass per target)
#   baseline_kernel   v0 single stream vs chunked, n=N_LEN
#                     -> overhead of the chunked decomposition
# then the theory z-test and the exact reproducibility checks.
#
# Note: this characterises the serial kernel. The T1 used for speedup is
# re-measured inside each scaling job (threads=1) on the same exclusive node.
#
#SBATCH --job-name=serial
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

REPS="${REPS:-5}"
N_LEN="${N_LEN:-4e9}"
N_TC="${N_TC:-1e9}"

targets() { grep -v '^#' "data/targets/$1.txt"; }

echo "== baseline_length (n=$N_LEN, reps=$REPS)"
out=$(csv_for baseline_length)
for t in $(targets lengths | head -7); do
    for s in $(seq 1 "$REPS"); do
        append_csv "$out" "$BIN/serial" -n "$N_LEN" -t "$t" -s "$s"
    done
done

echo "== baseline_ntargets (n=$N_TC, reps=$REPS)"
out=$(csv_for baseline_ntargets)
for k in 1 2 4 8 16; do
    spec=$(targets five_letter | head -"$k" | paste -sd, -)
    for s in $(seq 1 "$REPS"); do
        append_csv "$out" "$BIN/serial" -n "$N_TC" -t "$spec" -s "$s"
    done
done

echo "== baseline_kernel (n=$N_LEN, reps=$REPS)"
out=$(csv_for baseline_kernel)
for k in v0 chunked; do
    for s in $(seq 1 "$REPS"); do
        append_csv "$out" "$BIN/serial" -n "$N_LEN" -t romeo -s "$s" -k "$k"
    done
done

echo "== correctness"
python3 analysis/check_theory.py --bin "$BIN/serial"
python3 analysis/check_theory.py --bin "$BIN/serial" --kernel v0
python3 analysis/check_repro.py --bin "$BIN/serial"

echo "done: job $SLURM_JOB_ID"
