#!/bin/bash
# Submit the experiments of the phases implemented so far, in order.
# Run from the repo root on the login node:  bash slurm/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

probe=$(sbatch --parsable slurm/probe.sbatch)
echo "probe   $probe"
serial=$(sbatch --parsable --dependency=afterany:"$probe" slurm/serial.sbatch)
echo "serial  $serial"
echo "watch with: squeue --me ; logs in results/logs/"
