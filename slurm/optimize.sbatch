#!/bin/bash
# Phase 5: optimization before/after (chunked baseline vs fused kernel) and
# compiler-flag sensitivity.
#
#     sbatch slurm/optimize.sbatch
#
#  A. single core: both kernels x 1, 2, 4, 8, 16 five-letter targets, plus a
#     mixed-length set (a, oh, the, love, romeo, juliet, capulet, montague)
#  B. 32 threads (whole allocation): both kernels x 1 and 16 targets, so the
#     optimization is shown at full scale as well
#  C. compiler flags, single core, both kernels, 1 target:
#     -O2 | -O3 (default) | -O3 -march=znver4 | -O3 -march=znver4 -funroll-loops
#     (Zen 5 has AVX-512; GCC 12.3 has no znver5 target, znver4 is closest)
#
# Outputs: results/raw/optimize_1core/<id>.csv, optimize_32/<id>.csv,
#          optimize_flags/<id>.csv (column "flags" added)
#
#SBATCH --job-name=optimize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh

REPS="${REPS:-3}"
N1="${N1:-2e9}"
N32="${N32:-1.2e11}"
CORES=$(phys_cores)
export OMP_PLACES=cores OMP_PROC_BIND=close
five() { targets five_letter | head -"$1" | paste -sd, -; }
MIXED=$(targets lengths | head -8 | paste -sd, -)

echo "== A. single core, kernels x targets"
out=$(csv_for optimize_1core)
for rep in $(seq 1 "$REPS"); do
    for k in chunked fused; do
        for nt in 1 2 4 8 16; do
            append_csv "$out" "$BIN/serial" -n "$N1" -k "$k" -t "$(five "$nt")" -s "$rep"
        done
        append_csv "$out" "$BIN/serial" -n "$N1" -k "$k" -t "$MIXED" -s "$rep"
    done
done

echo "== B. $CORES threads, kernels x {1,16} targets"
out=$(csv_for optimize_32)
for rep in $(seq 1 "$REPS"); do
    for k in chunked fused; do
        for nt in 1 16; do
            OMP_NUM_THREADS=$CORES append_csv "$out" "$BIN/omp" -n "$N32" -k "$k" \
                -t "$(five "$nt")" -s "$rep"
        done
    done
done

echo "== C. compiler flags (single core)"
out=$(csv_for optimize_flags)
FLAG_SETS="${FLAG_SETS:--O2|-O3|-O3 -march=znver4|-O3 -march=znver4 -funroll-loops}"
IFS='|' read -r -a flag_sets <<< "$FLAG_SETS"
i=0
for flags in "${flag_sets[@]}"; do
    i=$((i + 1))
    fb="$BIN/flags$i"
    make -s BIN="$fb" CFLAGS="$flags -std=gnu11 -Wall -Wextra" "$fb/serial"
    tag=$(echo "$flags" | tr ' ' '_')
    for rep in $(seq 1 "$REPS"); do
        for k in chunked fused; do
            tmp=$(mktemp)
            "$fb/serial" -n "$N1" -k "$k" -t romeo -s "$rep" -H > "$tmp"
            if [ -s "$out" ]; then
                tail -n +2 "$tmp" | sed "s/\$/,$tag/" >> "$out"
            else
                sed "1s/\$/,flags/; 2,\$s/\$/,$tag/" "$tmp" >> "$out"
            fi
            rm -f "$tmp"
        done
    done
done
echo "done: job $SLURM_JOB_ID"
