#!/bin/bash
# Phase 0: record what a compute node looks like (cores, sockets, NUMA,
# memory, frequency policy, toolchain). Output: env/cluster/<partition>_<host>.txt
#
#     sbatch slurm/probe.sbatch                 # cpu partition
#     sbatch -p gpu --gres=gpu:1 slurm/probe.sbatch
#
#SBATCH --job-name=probe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=results/logs/%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source env/modules.sh

mkdir -p env/cluster
out="env/cluster/${SLURM_JOB_PARTITION}_$(hostname -s).txt"
section() { echo; echo "=== $*"; }
{
    echo "# probe job ${SLURM_JOB_ID} on $(hostname) at $(date -Iseconds)"
    section "sinfo (partitions: nodes, cpus/node, memory MB, gres)"
    sinfo -o "%P %D %c %m %G %l" 2>&1
    section "sinfo per node"
    sinfo -N -o "%N %P %c %m %G %T" 2>&1
    section "lscpu"
    lscpu
    section "numa"
    (numactl -H 2>/dev/null || lscpu | grep -i numa)
    section "memory"
    free -g
    section "cpu frequency policy"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
    cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null && echo "(intel no_turbo)"
    cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null && echo "(boost)"
    section "smt"
    cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo "n/a"
    section "toolchain"
    module -t list 2>&1
    echo "gcc:   $(command -v gcc)  $(gcc --version | head -1)"
    echo "mpicc: $MPICC  ($(command -v mpicc) on PATH)"
    "$MPICC" --showme 2>&1 || true
    echo "mpirun: $(command -v mpirun)  $(mpirun --version 2>&1 | head -1)"
    echo "srun mpi plugins: $(srun --mpi=list 2>&1 | tr '\n' ' ')"
    echo "python3: $(command -v python3)  $(python3 --version 2>&1)"
    section "profilers available"
    for t in perf likwid-perfctr nsys ncu nvc nvcc; do
        printf "%-15s %s\n" "$t" "$(command -v $t || echo '-')"
    done
    section "modules of interest"
    module -t avail 2>&1 | grep -iE "^(gcc|openmpi|intel|nvhpc|cuda|hpc-course|likwid|papi|score-p|extrae|paraver|python|scipy-bundle|matplotlib)/" | sort -u
    if command -v nvidia-smi >/dev/null; then
        section "gpu"
        nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
    fi
} > "$out" 2>&1

echo "wrote $out"
