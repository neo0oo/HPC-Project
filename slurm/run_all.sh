#!/bin/bash
# Submit the experiments in one go. Run from the repo root on the login node:
#
#     bash slurm/run_all.sh            phases 3-7 (smoke test first)
#     bash slurm/run_all.sh --all      also phases 0-2 (probe, serial, OpenMP)
#
# Every experiment depends on the smoke test (afterok): if smoke fails, the
# rest are cancelled automatically instead of running broken code. The QOS
# limit (64 CPUs / 1 GPU per user) makes Slurm run them one after another.
# Job ids go to results/logs/run_all_<date>.txt; when `squeue --me` is empty:
#     python3 analysis/summarize.py > results/summary.txt
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log="results/logs/run_all_$(date +%Y%m%d_%H%M%S).txt"

sub() { local name="$1"; shift; local id; id=$(sbatch --parsable "$@"); \
        printf "%-16s %s\n" "$name" "$id" | tee -a "$log"; echo "$id" > /dev/null; }

if [ "${1:-}" = "--all" ]; then
    sub probe slurm/probe.sbatch
    sub serial slurm/serial.sbatch
    sub omp_strong slurm/omp_strong.sbatch
    sub omp_weak slurm/omp_weak.sbatch
fi

smoke=$(sbatch --parsable slurm/smoke.sbatch)
printf "%-16s %s\n" smoke "$smoke" | tee -a "$log"
dep=(--dependency=afterok:"$smoke" --kill-on-invalid-dep=yes)

for s in mpi_strong mpi_weak mpi_2node hybrid optimize sensitivity profile \
         first_match_k first_match; do
    sub "$s" "${dep[@]}" "slurm/$s.sbatch"
done
sub gpu_ada "${dep[@]}" --gres=gpu:rtx6000ada:1 slurm/gpu.sbatch
sub gpu_blackwell "${dep[@]}" --gres=gpu:rtx6000bb:1 slurm/gpu.sbatch

echo
echo "submitted; ids in $log"
echo "watch: squeue --me     smoke log: results/logs/smoke_${smoke}.out"
