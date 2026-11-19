#!/bin/bash
# Phase 5: sensitivity of the OpenMP run to chunk size and loop schedule.
#
#     sbatch slurm/sensitivity.sbatch
#
# All cores of the allocation (32), fused kernel (fast, so scheduling
# overheads are not hidden behind a slow kernel), n fixed.
#   chunk = 2^12 ... 2^32 characters
#     tiny chunks:  more scheduling events (atomic counter of dynamic,1),
#                   more seeding work per character
#     huge chunks:  too few chunks for 32 threads -> load imbalance
#                   (busy_max / busy_min shows it directly)
#   schedule = static | dynamic,1 | guided
#
# Output: results/raw/sensitivity/<id>.csv (column "schedule" added)
#
#SBATCH --job-name=sensitivity
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

REPS="${REPS:-3}"
N="${N:-1.2e11}"
CORES=$(phys_cores)
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=$CORES

out=$(csv_for sensitivity)
echo "== chunk size x schedule, $CORES threads, n=$N"
for rep in $(seq 1 "$REPS"); do
    for e in 12 14 16 18 20 22 24 26 28 30 32; do
        for sched in static dynamic,1 guided; do
            tmp=$(mktemp)
            OMP_SCHEDULE=$sched "$BIN/omp" -n "$N" -k fused -t romeo -c "2^$e" \
                -s "$rep" -H > "$tmp"
            tag=${sched/,/_}
            if [ -s "$out" ]; then
                tail -n +2 "$tmp" | sed "s/\$/,$tag/" >> "$out"
            else
                sed "1s/\$/,schedule/; 2,\$s/\$/,$tag/" "$tmp" >> "$out"
            fi
            rm -f "$tmp"
        done
    done
done
echo "done: job $SLURM_JOB_ID"
