#!/bin/bash
# Phase 3: MPI + OpenMP hybrid, ranks x threads sweep on 2 nodes.
#
#     sbatch slurm/hybrid.sbatch
#
# Allocation: 2 nodes x 16 cores = 32 cores. Every configuration uses all
# 32 cores, split differently:
#   32 ranks x 1 thread, 16 x 2, 8 x 4, 4 x 8, 2 x 16 (one rank per node)
# OMP_PLACES=cores, OMP_PROC_BIND=close; each rank owns -c <threads> cores.
# Same total work, so the differences are the cost of more processes
# (startup, a bigger reduction tree) vs more threads (dynamic scheduling,
# shared caches), plus load imbalance from chunk granularity.
#
# Output: results/raw/hybrid/<id>.csv
#
#SBATCH --job-name=hybrid
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=500M
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N="${N:-1.2e11}"
REPS="${REPS:-3}"
TARGETS="${TARGETS:-romeo}"
export OMP_PLACES=cores OMP_PROC_BIND=close

out=$(csv_for hybrid)
echo "== hybrid sweep on 2 nodes x 16 cores, n=$N reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    for rt in 32x1 16x2 8x4 4x8 2x16; do
        r=${rt%x*}; t=${rt#*x}
        echo "rep $rep ranks $r threads $t"
        OMP_NUM_THREADS=$t append_csv "$out" $SRUN -N 2 -n "$r" \
            --ntasks-per-node=$((r / 2)) -c "$t" --cpu-bind=cores \
            "$BIN/hybrid" -n "$N" -t "$TARGETS" -s "$rep"
    done
done
echo "done: job $SLURM_JOB_ID"
