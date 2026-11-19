#!/bin/bash
# Smoke test (~5 min): every CPU program on small inputs, across 2 nodes.
# Gate for the long experiments: run_all.sh starts them only if this passes.
#
#     sbatch slurm/smoke.sbatch
#
# - full `make test` with MPI programs on 4 ranks over 2 nodes
#   (exact counts of serial / fused / omp / mpi / hybrid, first_match
#   exactness across configs and its mean wait vs Conway)
# - prof (timing breakdown) and, if the PAPI module works, hardware counters
#
#SBATCH --job-name=smoke
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=500M
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

echo "== make test (MPI: 4 ranks on 2 nodes)"
make -s BIN="$BIN" test LAUNCH="$SRUN -n 4 --ntasks-per-node=2"

echo "== srun layouts used by the hybrid / first_match jobs"
OMP_NUM_THREADS=8 OMP_PLACES=cores $SRUN -N 2 -n 2 --ntasks-per-node=1 -c 8 \
    --cpu-bind=cores "$BIN/hybrid" -n 2e9 -t romeo -H
OMP_NUM_THREADS=4 $SRUN -N 1 -n 1 -c 4 "$BIN/first_match" -t love -n 1e12 -c 2^16 -H

echo "== prof (timing breakdown)"
for k in gen chunked fused; do
    "$BIN/prof" -n 2e8 -k "$k" -t romeo
done

echo "== PAPI (optional)"
(
    module load PAPI/7.1.0-GCCcore-13.2.0 >/dev/null 2>&1 &&
    make -s BIN="$BIN" CC=gcc papi &&
    "$BIN/prof_papi" -n 2e8 -k fused -t romeo -H
) || echo "PAPI not usable (fine: prof timing breakdown is the fallback)"
echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"

echo "SMOKE PASSED: job $SLURM_JOB_ID"
