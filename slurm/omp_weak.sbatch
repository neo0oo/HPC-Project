#!/bin/bash
# Phase 2: OpenMP weak scaling + thread binding, one node.
#
#     sbatch slurm/omp_weak.sbatch
#
# CPU budget: 64 CPUs per user (QOS student), see omp_strong.sbatch.
#
# Work per thread is fixed (N_PER_THREAD characters), so ideal time is flat.
# Every thread count is run with two bindings (OMP_PLACES=cores):
#   close   fill socket 0 first (cores next to each other)
#   spread  spread threads over both sockets
# Runs are ~17 s each, long enough for the CPU sampler to show how the real
# clock frequency drops as more cores are active.
#
# Outputs:
#   results/raw/omp_weak/<bind>/<id>.csv       program CSV (bind = close|spread)
#   results/raw/omp_weak/<bind>/<id>_cpu.csv   measured MHz + utilization
#
#SBATCH --job-name=omp_weak
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N_PER_THREAD="${N_PER_THREAD:-1e10}"   # ~17 s per run at 5.84e8 chars/s
REPS="${REPS:-3}"
TARGETS="${TARGETS:-romeo}"
CORES=$(phys_cores)
per=$(python3 -c "print(int(float('$N_PER_THREAD')))")

export OMP_PLACES=cores

echo "== weak scaling n/thread=$N_PER_THREAD targets=$TARGETS cores=$CORES reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    for t in $(pow2_upto "$CORES"); do
        for bind in close spread; do
            n=$((per * t))
            out=$(csv_for "omp_weak/$bind")
            cpu="${out%.csv}_cpu.csv"
            echo "rep $rep threads $t bind $bind n $n"
            OMP_PROC_BIND=$bind OMP_NUM_THREADS=$t run_sampled "$cpu" \
                "rep=$rep;threads=$t;bind=$bind" \
                "$out" "$BIN/omp" -n "$n" -t "$TARGETS" -s "$rep"
        done
    done
done

echo "done: job $SLURM_JOB_ID"
