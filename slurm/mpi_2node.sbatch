#!/bin/bash
# Phase 3: one node vs two nodes at equal core count (pure MPI).
#
#     sbatch slurm/mpi_2node.sbatch
#
# Allocation: 2 nodes x 16 cores = 32 cores (64 CPUs with SMT = QOS limit).
#   16 ranks on 1 node        vs  16 ranks split 8 + 8 over 2 nodes
#   32 ranks over 2 nodes (16 + 16)   (compare with mpi_strong 32 on 1 node)
# Same work, so differences come from the network (inter-node reduction,
# comm_s) and from the nodes themselves (load from other users, clocks).
#
# Output: results/raw/mpi_2node/<id>.csv  (a "layout" column is added)
#
#SBATCH --job-name=mpi_2node
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=500M
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

N="${N:-1.2e11}"
REPS="${REPS:-3}"
TARGETS="${TARGETS:-romeo}"

out=$(csv_for mpi_2node)
run() { # run <layout> <srun options...>
    local layout="$1"; shift
    local tmp; tmp=$(mktemp)
    $SRUN "$@" --cpu-bind=cores "$BIN/mpi" -n "$N" -t "$TARGETS" -s "$rep" -H > "$tmp"
    if [ -s "$out" ]; then
        tail -n +2 "$tmp" | sed "s/\$/,$layout/" >> "$out"
    else
        sed "1s/\$/,layout/; 2,\$s/\$/,$layout/" "$tmp" >> "$out"
    fi
    rm -f "$tmp"
}

echo "== 1 node vs 2 nodes, n=$N reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    echo "rep $rep"
    run 16x1node -N 1 -n 16
    run 8+8_2nodes -N 2 -n 16 --ntasks-per-node=8
    run 16+16_2nodes -N 2 -n 32 --ntasks-per-node=16
done
echo "done: job $SLURM_JOB_ID"
