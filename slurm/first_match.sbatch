#!/bin/bash
# Phase 4: time-to-solution. Type until the target first appears.
#
#     sbatch slurm/first_match.sbatch
#
# Allocation: 2 nodes x 16 cores, hybrid 2 ranks x 16 threads (fused kernel,
# chunk 2^20 ~ 1 ms of work, stop check every K=16 chunks per thread).
#
#  A. theory     30 seeds each for romeo (5), juliet (6), capulet (7),
#                montague (8): distribution of first_pos vs Conway's mean
#                (waiting time ~ geometric -> plot ECDF vs 1 - exp(-x/E))
#  B. scaling    capulet, 20 seeds, on 1, 2, 4, 8, 16 cores (1 rank) and
#                32 cores (2 ranks): median time-to-solution vs cores
#  C. headline   "wherefore" (9 letters, mean wait 7.6e12 chars) on all
#                32 cores, 5 seeds, async check
#
# Output: results/raw/first_match/<id>.csv  (one row per run, FM_HEADER)
#
#SBATCH --job-name=first_match
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=500M
#SBATCH --time=02:30:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

export OMP_PLACES=cores OMP_PROC_BIND=close
FM="-n 1e16 -c 2^20 -K 16 -k fused"
A_TARGETS="${A_TARGETS:-romeo juliet capulet montague}"
A_SEEDS="${A_SEEDS:-30}"
B_TARGET="${B_TARGET:-capulet}"
B_SEEDS="${B_SEEDS:-20}"
C_TARGET="${C_TARGET:-wherefore}"
C_SEEDS="${C_SEEDS:-5}"
out=$(csv_for first_match)

all32() { OMP_NUM_THREADS=16 append_csv "$out" $SRUN -N 2 -n 2 \
              --ntasks-per-node=1 -c 16 --cpu-bind=cores "$BIN/first_match" "$@"; }

echo "== A. distribution vs theory (2 x 16 cores)"
for t in $A_TARGETS; do
    for s in $(seq 1 "$A_SEEDS"); do
        all32 $FM -t "$t" -s "$s"
    done
done

echo "== B. time-to-solution vs cores ($B_TARGET)"
for s in $(seq 101 $((100 + B_SEEDS))); do
    for c in 1 2 4 8 16; do
        OMP_NUM_THREADS=$c append_csv "$out" $SRUN -N 1 -n 1 -c "$c" \
            --cpu-bind=cores "$BIN/first_match" $FM -t "$B_TARGET" -s "$s"
    done
    all32 $FM -t "$B_TARGET" -s "$s"
done

echo "== C. headline: $C_TARGET on 32 cores"
for s in $(seq 1 "$C_SEEDS"); do
    all32 -n 1e16 -c 2^20 -K 16 -k fused -m async -t "$C_TARGET" -s "$s"
    tail -1 "$out"
done
echo "done: job $SLURM_JOB_ID"
