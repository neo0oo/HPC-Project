#!/bin/bash
# Phase 3: pure MPI strong scaling on one node (fixed total work).
#
#     sbatch slurm/mpi_strong.sbatch
#
# ranks = 1, 2, 4, ..., 32 (one rank per physical core: --hint=nomultithread,
# --cpu-bind=cores). 32 cores x 2 hardware threads = the 64-CPU QOS limit.
# Compare with omp_strong at the same core counts: same kernel, same chunks,
# so any difference is process vs thread overhead (startup, reduction).
#
# Output: results/raw/mpi_strong/<id>.csv (comm_s = final reduction time)
#
#SBATCH --job-name=mpi_strong
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=500M
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N="${N:-1.2e11}"
REPS="${REPS:-3}"
TARGETS="${TARGETS:-romeo}"

out=$(csv_for mpi_strong)
echo "== MPI strong scaling n=$N reps=$REPS tasks=$SLURM_NTASKS"
for rep in $(seq 1 "$REPS"); do
    for p in $(pow2_upto "$SLURM_NTASKS"); do
        echo "rep $rep ranks $p"
        append_csv "$out" $SRUN -n "$p" --cpu-bind=cores \
            "$BIN/mpi" -n "$N" -t "$TARGETS" -s "$rep"
    done
done
echo "done: job $SLURM_JOB_ID"
