#!/bin/bash
# Phase 6: GPU (CUDA) + OpenACC exploration, one GPU.
#
#     sbatch slurm/gpu.sbatch                                   # any GPU
#     sbatch --gres=gpu:rtx6000ada:1 slurm/gpu.sbatch           # Ada (haskell)
#     sbatch --gres=gpu:rtx6000bb:1  slurm/gpu.sbatch           # Blackwell
#
#  0. correctness: CUDA counts must equal ./serial exactly (same chunks)
#  1. block size sweep (64..1024 threads per block)
#  2. chunk size sweep (2^10..2^22 characters per thread task)
#  3. number of targets (1, 4, 16)
#  4. headline: count "wherefore" (9 letters) and "friar john" (10) over as
#     many characters as ~HEADLINE_S seconds allow at the measured rate
#  5. OpenACC (nvhpc): same kernel, -acc=gpu and -acc=multicore; exactness
#     and speed vs the hand-written CUDA kernel / OpenMP
#  6. Nsight Systems / Compute summaries if the tools are on PATH
#
# Outputs: results/raw/gpu_*/<id>.csv, Nsight reports in results/logs/
#
#SBATCH --job-name=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/%x_%j.out

source slurm/common.sh
module load hpc-course-cuda/12.8.1
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version,clocks.max.sm \
    --format=csv | tee -a "results/logs/${JOBTAG}.env"
make -s BIN="$BIN" cuda
CU="$BIN/monkey_cuda"
REPS="${REPS:-3}"
N="${N:-1e12}"
HEADLINE_S="${HEADLINE_S:-300}"

echo "== 0. CUDA == serial (exact counts)"
python3 analysis/check_repro.py --bin "$BIN/serial" --compare "$CU"
python3 analysis/check_theory.py --bin "$CU" --kernel fused -n 2e10

echo "== 1. block size"
out=$(csv_for gpu_block)
for rep in $(seq 1 "$REPS"); do
    for b in 64 128 256 512 1024; do
        CUDA_BLOCK=$b append_csv "$out" "$CU" -n "$N" -t romeo -c 2^16 -s "$rep"
    done
done

echo "== 2. chunk size"
out=$(csv_for gpu_chunk)
for rep in $(seq 1 "$REPS"); do
    for e in 10 12 14 16 18 20 22; do
        append_csv "$out" "$CU" -n "$N" -t romeo -c "2^$e" -s "$rep"
    done
done

echo "== 3. number of targets"
out=$(csv_for gpu_targets)
for rep in $(seq 1 "$REPS"); do
    for nt in 1 4 16; do
        t=$(targets five_letter | head -"$nt" | paste -sd, -)
        append_csv "$out" "$CU" -n "$N" -t "$t" -c 2^16 -s "$rep"
    done
done

echo "== 4. headline counts"
rate=$("$CU" -n "$N" -t romeo -c 2^16 -s 99 | cut -d, -f13)
nh=$(python3 -c "print(f'{float($rate) * $HEADLINE_S:.3e}')")
echo "rate=$rate chars/s -> n=$nh per headline run"
out=$(csv_for gpu_headline)
append_csv "$out" "$CU" -n "$nh" -t wherefore,friar_john -c 2^18 -s 1
cat "$out"

echo "== 5. OpenACC (nvhpc) vs OpenMP on the same $SLURM_CPUS_PER_TASK CPU cores"
out=$(csv_for gpu_openacc)
for rep in $(seq 1 "$REPS"); do   # OpenMP reference (GCC toolchain still loaded)
    OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores \
        append_csv "$out" "$BIN/omp" -n 4e10 -k fused -t romeo -c 2^20 -s "$rep"
done
(
    module purge >/dev/null 2>&1
    module load nvhpc/25.7 &&
    { make -s BIN="$BIN" acc ACCGPU="-acc=gpu -gpu=ccnative" ||
      make -s BIN="$BIN" acc; } &&
    python3 analysis/check_repro.py --bin "$BIN/serial" --compare "$BIN/acc_gpu" &&
    for rep in $(seq 1 "$REPS"); do
        append_csv "$out" "$BIN/acc_gpu" -n "$N" -t romeo -c 2^16 -s "$rep"
        ACC_NUM_CORES=$SLURM_CPUS_PER_TASK \
            append_csv "$out" "$BIN/acc_multicore" -n 4e10 -t romeo -c 2^20 -s "$rep"
    done
) || echo "OpenACC step failed (recorded as a negative result)"

echo "== 6. Nsight (optional)"
if command -v nsys >/dev/null; then
    nsys profile -o "results/logs/${JOBTAG}_nsys" --force-overwrite true \
        --stats=true "$CU" -n 1e11 -t romeo -c 2^16 2>&1 | tail -40
else
    echo "nsys not on PATH"
fi
if command -v ncu >/dev/null; then
    ncu --section SpeedOfLight --section Occupancy --section LaunchStats \
        "$CU" -n 1e10 -t romeo -c 2^16 2>&1 | tail -60 ||
        echo "ncu failed (often needs GPU counter permissions)"
else
    echo "ncu not on PATH"
fi
echo "done: job $SLURM_JOB_ID"
