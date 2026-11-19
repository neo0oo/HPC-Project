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
#   run_sampled <freq.csv> <label> <csv> <cmd...>
#                               append_csv while sampling CPU MHz/utilization
#   phys_cores / logical_cpus   cores / hardware threads this job may use
#   pow2_upto <n>               1 2 4 ... <= n, plus n itself
#   SRUN                        MPI launcher: srun with the PMIx plugin
#   targets <list>              targets from data/targets/<list>.txt

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

# append_csv_col <csv> <column> <value> <command...>: like append_csv, plus one
# extra trailing column (e.g. a setting passed by environment variable).
append_csv_col() {
    local out="$1" col="$2" val="$3"; shift 3
    if [ -s "$out" ]; then
        "$@" | sed "s/\$/,$val/" >> "$out"
    else
        "$@" -H | sed "1s/\$/,$col/; 2,\$s/\$/,$val/" >> "$out"
    fi
}

# Physical cores (distinct socket:core pairs) and hardware threads in our
# affinity mask. Slurm decides these from --cpus-per-task / --hint.
phys_cores() {
    python3 -c '
import os
if not hasattr(os, "sched_getaffinity"):  # not Linux (local dry runs)
    print(os.cpu_count()); raise SystemExit
cpus = os.sched_getaffinity(0)
def rd(c, f): return open(f"/sys/devices/system/cpu/cpu{c}/topology/{f}").read().strip()
print(len({(rd(c, "physical_package_id"), rd(c, "core_id")) for c in cpus}))'
}
logical_cpus() { python3 -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())'; }

pow2_upto() {
    local n="$1" p=1 out=""
    while [ "$p" -lt "$n" ]; do out="$out $p"; p=$((p * 2)); done
    echo "$out $n"
}

# run_sampled <freq.csv> <label> <out.csv> <cmd...>
run_sampled() {
    local freq="$1" label="$2" out="$3"; shift 3
    python3 slurm/cpu_sampler.py --out "$freq" --label "$label" &
    local spid=$!
    append_csv "$out" "$@"
    kill -TERM "$spid" 2>/dev/null || true
    wait "$spid" 2>/dev/null || true
}

# MPI launcher. OpenMPI 4.1.5 (EESSI) is started by Slurm through PMIx.
SRUN="${SRUN:-srun --mpi=pmix}"

# Transport. The default (UCX + shared memory) segfaults on haskell with 8+
# ranks per node (job 31232/31233: shared-memory backing file open() ENOENT,
# then a UCX crash). Use the plain ob1/TCP path instead. Our only messages are
# one MPI_Reduce per run (and one small Allreduce per first_match round), so
# transport speed does not affect the results; comm_s still measures it.
# MPI_TRANSPORT=default restores Open MPI's own choice.
if [ "${MPI_TRANSPORT:-tcp}" = tcp ]; then
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_osc=pt2pt
    export OMPI_MCA_btl_tcp_if_exclude=lo,docker0
fi

targets() { grep -v '^#' "data/targets/$1.txt"; }

record_env
echo "cores=$(phys_cores) hw_threads=$(logical_cpus)" | tee -a "results/logs/${JOBTAG}.env"
