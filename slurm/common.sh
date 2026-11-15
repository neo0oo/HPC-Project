# Sourced by every .sbatch script. Must be submitted from the repo root:
#     sbatch slurm/<script>.sbatch
#
# Provides:
#   BIN        per-job build directory with freshly compiled binaries
#   RAW        results/raw (CSV output)
#   JOBTAG     <job-name>_<job-id>, used for every file this job writes
#   record_env writes results/logs/$JOBTAG.env (provenance of this run)
#   csv_for <experiment>        prints results/raw/<experiment>/<job-id>.csv
#   append_csv <file> <cmd...>  runs cmd, appends CSV rows, header once

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repo root}"
[ -f Makefile ] && [ -d src ] || { echo "error: submit from the repo root" >&2; exit 2; }

source env/modules.sh

JOBTAG="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
BIN="build/${SLURM_JOB_ID}"
RAW="results/raw"
mkdir -p "$RAW" results/logs

# Compile on the compute node, with the loaded toolchain, into a private
# directory (no races between jobs, and the binary matches this CPU).
make -s BIN="$BIN" all

record_env() {
    local f="results/logs/${JOBTAG}.env"
    {
        echo "# provenance for job ${SLURM_JOB_ID}"
        echo "date=$(date -Iseconds)"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_dirty=$(git diff --quiet HEAD -- 2>/dev/null && echo no || echo yes)"
        echo "host=$(hostname)"
        echo "nodelist=${SLURM_JOB_NODELIST:-}"
        echo "partition=${SLURM_JOB_PARTITION:-}"
        echo "ntasks=${SLURM_NTASKS:-} cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
        echo "cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
        echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
        echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-} OMP_PROC_BIND=${OMP_PROC_BIND:-} OMP_PLACES=${OMP_PLACES:-}"
        echo "--- make flags"
        make -s BIN="$BIN" flags
        echo "--- modules"
        module -t list 2>&1
    } > "$f"
}

# One CSV per experiment per job, so reruns never mix or overwrite.
csv_for() {
    mkdir -p "$RAW/$1"
    echo "$RAW/$1/${SLURM_JOB_ID}.csv"
}

# append_csv <file> <command...>: run command, keep its header only if the
# file is new, append rows. Commands must accept -H.
append_csv() {
    local out="$1"; shift
    if [ -s "$out" ]; then
        "$@" >> "$out"
    else
        "$@" -H >> "$out"
    fi
}

record_env
