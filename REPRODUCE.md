# Reproducing our results

All commands are run from the repository root on the SciTech login node. Every job compiles its own binary on the compute node (into `build/<job-id>/`), so no build step is needed on the login node.

## 0. Get the code

```bash
git clone https://github.com/neo0oo/HPC-Project.git
cd HPC-Project
git checkout <tag used in the report>
```

## 1. Toolchain

`env/modules.sh` loads `OpenMPI/4.1.5-GCC-12.3.0` (GCC 12.3.0). Compiler flags are in the `Makefile` (`make flags` prints them). Every job also stores its exact flags, modules, commit and CPU model in `results/logs/<job>_<id>.env`.

Python is 3.11.4 on the cluster. The scripts only need the standard library, except plotting, which needs matplotlib 3.7.2 (`module load matplotlib/3.7.2-gfbf-2023a` on the cluster, or `pip install -r env/requirements.txt` elsewhere).

## 2. Phase 0: describe the cluster

```bash
sbatch slurm/probe.sbatch
sbatch -p gpu --gres=gpu:1 slurm/probe.sbatch     # optional, GPU node
```

Output: `env/cluster/<partition>_<host>.txt`.

## 3. Phase 1: serial baseline

```bash
sbatch slurm/serial.sbatch
```

Defaults: 5 repetitions (seeds 1–5), `N_LEN=4e9` for the length sweep, `N_TC=1e9` for the number-of-targets sweep. All three can be overridden with environment variables, e.g. `REPS=3 sbatch slurm/serial.sbatch`.

Output:
- `results/raw/baseline_length/<id>.csv`
- `results/raw/baseline_ntargets/<id>.csv`
- `results/raw/baseline_kernel/<id>.csv`
- the correctness checks at the end of `results/logs/serial_<id>.out`.

Or submit everything implemented so far: `bash slurm/run_all.sh`.

## 4. Phase 2: OpenMP scaling (one node)

```bash
sbatch slurm/omp_strong.sbatch
sbatch --dependency=singleton slurm/omp_weak.sbatch
```

Both jobs request 64 CPUs, the per-user limit of the class QOS `student`; that is 32 physical cores with SMT. They detect the physical core count from their affinity mask and run one thread per core (`OMP_PLACES=cores`), plus one SMT point. Each job first checks that the OpenMP counts are bit-identical to serial.
- **Strong scaling:** n = 1.2e11 fixed; threads 1, 2, 4, … up to the detected core count, plus one SMT point.
- **Weak scaling:** 1e10 characters per thread; close vs spread binding.

Each run also records the measured clock frequency and utilization of the busy cores (`<id>_cpu.csv`).

Summaries and plots:

```bash
python3 analysis/scaling.py strong results/raw/omp_strong/<id>.csv --plot results/figures/omp_strong.png
python3 analysis/scaling.py weak results/raw/omp_weak/close/<id>.csv results/raw/omp_weak/spread/<id>.csv --plot results/figures/omp_weak.png
```

## 5. Phases 3–7: everything else, in one submission

```bash
bash slurm/run_all.sh
```

This submits `smoke.sbatch` first; every other job depends on it (`afterok`), so a failing smoke test cancels the rest. The QOS (64 CPUs and 1 GPU per user) makes Slurm run them one after another, roughly 4–5 hours in total. The job ids are saved in `results/logs/run_all_<date>.txt`.

What each job measures:
- **`smoke`:** `make test` with MPI on 4 ranks over 2 nodes. Exact counts for serial, fused, omp, mpi and hybrid; first_match exactness and its mean wait vs Conway; PAPI availability.
- **`mpi_strong` / `mpi_weak`:** pure MPI, 1 → 32 ranks on one node.
- **`mpi_2node`:** 16 ranks on 1 node vs 8 + 8 on 2 nodes, and 32 ranks on 2 nodes.
- **`hybrid`:** ranks × threads = 32×1, 16×2, 8×4, 4×8, 2×16 on 2 nodes.
- **`first_match`:** waiting-time distribution vs theory (romeo, juliet, capulet, montague, 30 seeds each); time-to-solution vs cores (capulet); "wherefore" on 32 cores.
- **`first_match_k`:** stop-check interval K and sync vs async (`MPI_Iallreduce`) trade-off.
- **`optimize`:** baseline vs fused kernel (1 core and 32 threads, 1–16 targets) and compiler flags.
- **`sensitivity`:** chunk size 2^12–2^32 × OpenMP schedule.
- **`profile`:** generate-only vs kernels (timing breakdown), plus PAPI counters if available.
- **`gpu`:** run twice, on an RTX 6000 Ada and on an RTX PRO 6000 Blackwell. CUDA exactness vs serial, then block size, chunk size and target-count sweeps, a headline count of "wherefore" and "friar john", OpenACC (nvhpc) gpu and multicore builds, and Nsight if available.

Summary of everything (any time; sections without data are skipped):

```bash
python3 analysis/summarize.py > results/summary.txt
module load matplotlib/3.7.2-gfbf-2023a && python3 analysis/summarize.py --plots results/figures
python3 analysis/figures.py          # the other report figures (needs matplotlib)
```

Slurm accounting for every job that has a provenance file (elapsed time, CPU time, peak memory), run on the cluster:

```bash
ids=$(ls results/logs/*.env | sed -E 's/.*_([0-9]+)\.env/\1/' | sort -un | paste -sd, -)
sacct -j "$ids" --format=JobID,JobName,Partition,NodeList,AllocCPUS,Elapsed,TotalCPU,MaxRSS,State > results/logs/sacct.txt
```

Local testing without MPI or GPUs: `make LOCAL_MPI_STUB=1 test` (single-rank MPI stand-in, see `src/mpi_stub/`).

## Job commits (`git_commit` in `results/logs/*.env`)

Every job records the commit it was built from. Most jobs ran in the cluster's clone, which cannot push to GitHub, so the commits they record were made there. Each of those commits is the GitHub commit listed below plus the Phase 0–1 result files (julia probe, serial job 30127). Those files are already on `main` with the same content. In `src/`, `slurm/`, `analysis/` and the `Makefile`, the code is byte-for-byte the same as the GitHub commit. The original commits are on GitHub as tags `job-<sha>`.

- `1a8a5b26f9f4` (on GitHub): serial 30127
- `44c3a4a9a5a7` = code of `9309139`: omp_strong 31203, omp_weak 31204
- `40b34f0f15cf` = code of `17f49b5`: smoke 31229, mpi_strong 31230, mpi_weak 31231, mpi_2node 31232, hybrid 31233, optimize 31234, sensitivity 31235, profile 31236
- `567485d81378` = code of `6df967c`: mpi_2node 31338, hybrid 31339, first_match_k 31340, first_match 31341, gpu 31342, gpu 31343
- `27de7a6681a9` = code of `879031e`: gpu 31621, gpu 31622
- `c44f7f12a28f` = code of `b051e0d`: gpu 32119
- `48e5ea0b5800` = code of `a6dcdd9`: gpu 32120
- `17ed3f143f25` = code of `e1d6ee9`: optimize 32267, profile 32268

To check one: `git fetch origin --tags && git diff --stat 40b34f0f15cf 17f49b5 -- src slurm analysis Makefile` prints nothing.

## Random seeds

Seeds are explicit (`-s`). Chunk i of a run with seed s is generated by xoshiro256** seeded from splitmix64 outputs 4i..4i+3 starting at s, so a given (n, chunk, seed, targets) gives identical counts on any hardware and any number of threads or ranks.
