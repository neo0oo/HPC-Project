#!/bin/bash
# Phase 3: pure MPI weak scaling on one node (fixed work per rank).
#
#     sbatch slurm/mpi_weak.sbatch
#
# n = N_PER_RANK x ranks, ranks = 1, 2, 4, ..., 32. Ideal: flat time.
#
# Output: results/raw/mpi_weak/<id>.csv
#
#SBATCH --job-name=mpi_weak
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=500M
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N_PER_RANK="${N_PER_RANK:-1e10}"
REPS="${REPS:-3}"
TARGETS="${TARGETS:-romeo}"
per=$(python3 -c "print(int(float('$N_PER_RANK')))")

out=$(csv_for mpi_weak)
echo "== MPI weak scaling n/rank=$N_PER_RANK reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    for p in $(pow2_upto "$SLURM_NTASKS"); do
        echo "rep $rep ranks $p"
        append_csv "$out" $SRUN -n "$p" --cpu-bind=cores \
            "$BIN/mpi" -n $((per * p)) -t "$TARGETS" -s "$rep"
    done
done
echo "done: job $SLURM_JOB_ID"
