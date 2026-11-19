#!/bin/bash
# Phase 2: OpenMP strong scaling on one node (fixed total work).
#
#     sbatch slurm/omp_strong.sbatch
#     sbatch --cpus-per-task=128 slurm/omp_strong.sbatch   # if the QOS allows
#
# CPU budget: the class QOS "student" allows 64 CPUs per user, and Slurm counts
# hardware threads, so 64 CPUs is normally 32 physical cores x 2 SMT threads.
# The script detects the real core count from its affinity mask
# (phys_cores) and scales up to it; with 2 threads per core it adds an SMT
# point. Ask for QOS student-128cpu (cpu=128) or a reservation to go further.
#
# threads = 1, 2, 4, ..., cores (one thread per physical core, close
# binding), REPS repetitions interleaved (rep is the outer loop so slow drift
# affects every thread count equally). If Slurm gave us both hardware threads
# of each core, one extra point uses SMT (2 threads per core).
#
# T1 for speedup is the threads=1 run of THIS job (same node, same build).
#
# Outputs:
#   results/raw/omp_strong/<id>.csv        program CSV (one row per run)
#   results/raw/omp_strong/<id>_cpu.csv    measured MHz + utilization per run
#
#SBATCH --job-name=omp_strong
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N="${N:-1.2e11}"          # ~205 s on 1 core at 5.84e8 chars/s
REPS="${REPS:-3}"
TARGETS="${TARGETS:-romeo}"
CORES=$(phys_cores)
HW=$(logical_cpus)

export OMP_PLACES=cores OMP_PROC_BIND=close

echo "== exact-count check vs serial (1, 7, $CORES threads)"
for t in 1 7 "$CORES"; do
    OMP_NUM_THREADS=$t python3 analysis/check_repro.py \
        --bin "$BIN/serial" --compare "$BIN/omp"
done

out=$(csv_for omp_strong)
cpu="${out%.csv}_cpu.csv"
echo "== strong scaling n=$N targets=$TARGETS cores=$CORES hw=$HW reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    for t in $(pow2_upto "$CORES"); do
        echo "rep $rep threads $t"
        OMP_NUM_THREADS=$t run_sampled "$cpu" "rep=$rep;threads=$t;places=cores" \
            "$out" "$BIN/omp" -n "$N" -t "$TARGETS" -s "$rep"
    done
    if [ "$HW" -ge $((2 * CORES)) ]; then
        echo "rep $rep threads $HW (SMT)"
        OMP_PLACES=threads OMP_NUM_THREADS=$HW run_sampled "$cpu" \
            "rep=$rep;threads=$HW;places=threads" \
            "$out" "$BIN/omp" -n "$N" -t "$TARGETS" -s "$rep"
    fi
done

echo "done: job $SLURM_JOB_ID"
