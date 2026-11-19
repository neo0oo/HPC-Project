#!/bin/bash
# Phase 4/5: communication trade-off of the stop check, sync vs async.
#
#     sbatch slurm/first_match_k.sbatch
#
# 2 ranks x 16 threads on 2 nodes, target montague (mean wait 2.8e11 chars),
# chunk 2^20 (~1 ms of work). Every K chunks per thread all ranks agree on
# whether anyone has found the target:
#   small K  -> many MPI_Allreduce + thread barriers (communication and
#               synchronization overhead, measured as comm_s)
#   large K  -> after the hit, everybody finishes a long round (wasted work,
#               measured as wasted_frac)
#   async    -> MPI_Iallreduce overlapped with the next round: hides the
#               communication, costs one extra round of wasted work
# Same seeds for every (K, mode), so first_pos is identical and only time,
# comm_s and wasted_frac change.
#
# Output: results/raw/first_match_k/<id>.csv
#
#SBATCH --job-name=first_match_k
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=500M
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=16
SEEDS="${SEEDS:-8}"
TARGET="${TARGET:-montague}"
out=$(csv_for first_match_k)

echo "== K sweep, sync vs async ($TARGET, $SEEDS seeds)"
for s in $(seq 1 "$SEEDS"); do
    for K in 1 4 16 64 256; do
        for m in sync async; do
            append_csv "$out" $SRUN -N 2 -n 2 --ntasks-per-node=1 -c 16 \
                --cpu-bind=cores "$BIN/first_match" -n 1e16 -c 2^20 \
                -k fused -t "$TARGET" -K "$K" -m "$m" -s "$s"
        done
    done
done
echo "done: job $SLURM_JOB_ID"
