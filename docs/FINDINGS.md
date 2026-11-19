# Findings log

Running record of measured results, with job IDs, for the report.

## Phase 0: cluster (probe job 30126, 2026-11-15)

- **Nodes:** haskell (2× RTX 6000 Ada) and julia (1× RTX PRO 6000 Blackwell) at the time of the probe. fortran (RTX PRO 6000 Blackwell) joined later and ran many of our jobs. All three have the same CPUs. The nodes are shared between the cpu, gpu and interactive partitions. Time limits: cpu 7 days, gpu 3 days, interactive 8 hours.
- **Each node:**
  - 2× AMD EPYC 9745 (Zen 5): 128 cores per socket, so 256 cores and 512 hardware threads (SMT on)
  - 2 NUMA nodes
  - L2 1 MiB per core, L3 512 MiB total
  - ~400–750 GB RAM
  - governor `performance`, boost on, max 3.7 GHz
- **Toolchain:** OpenMPI 4.1.5 / GCC 12.3.0 from EESSI (the stack is built for zen2), Python 3.11.4. `srun --mpi` supports pmix and pmi2.
- **Profiling:** `perf` and LIKWID are absent. The `PAPI/7.1.0` module provides hardware counters. Tracing modules: OTF2, Cube, OPARI2, Paraver.
- **Other:** `nvhpc/25.7` provides OpenACC via `nvc`. `hpc-course-cuda/12.8.1` provides CUDA.

- **Per-user limits (QOS `student`, account hpcclass):** at most 64 CPUs (hardware threads) and 1 GPU per user at a time. Other QOS levels exist: `student-128cpu` (128 CPUs), `student-2gpu`, `normal` (128 CPUs, 2 GPUs). Jobs over the limit wait forever with `QOSMaxCpuPerUserLimit`.

Implications:
- With our QOS, scaling is 1 → 32 physical cores (64 CPUs with SMT) per job. A node has 256 cores, so the per-user QOS is the limit, not the hardware.
  - Going further needs QOS `student-128cpu` or a reservation from the professor. Each group member also has their own 64-CPU budget, so independent experiments can run in parallel under different accounts.
  - Multi-node MPI fits in the budget, e.g. 2 nodes × 32 CPUs.
- Zen 5 has AVX-512, but GCC 12.3 has no `znver5` target. Use `-march=znver4` or `-mavx512f` for the SIMD optimization.

## Phase 1: serial baseline (job 30127, julia, 1 core, 5 reps each)

- **Single-core speed:** 5.84e8 chars/s with one target, with run-to-run spread below 0.5%.
- **Target length does not matter:** L = 1..7 all run at 5.83–5.84e8 chars/s. The kernel cost does not depend on L.
- **Chunked decomposition is free:** v0 runs at 5.82e8 and chunked at 5.84e8, which is within noise.
- **Cost per target:** time per character fits almost perfectly to
  `t(k) = 0.750 + 0.959·k  ns/char`  (k = number of targets, error < 0.01 ns)
  - About 0.75 ns/char goes to generating the random character. That is roughly 2.8 cycles at 3.7 GHz.
  - About 0.96 ns/char goes to matching, per target. That is roughly 3.5 cycles, which matches the loop-carried `w*27 - out*27^L + c` dependency chain (a 3-cycle multiply plus add).
  - With 1 target, matching is 56% of the time. With 16 targets it is 95%.
  - This is the bottleneck hypothesis for Phase 5, to confirm with PAPI. Fusing the per-target passes and replacing the multiply chain with a 5-bit shift register should go after exactly this.
- **Correctness:** theory z-tests pass for both kernels, as do the reproducibility checks. Counts are bit-identical to the same run on an Apple M3, so results reproduce across machines.

Projected time to find each target (mean wait at 5.84e8 chars/s per core, assuming perfect scaling):
- "wherefore" (9 chars): ~51 s on 256 cores, ~26 s on 512
- a 10-char target: ~23 min on 256 cores, ~11 min on 512
- an 11-char target: ~10 h on 256 cores, ~5 h on 512


## Phases 2–7: all results (jobs 31203–32268, 2026-11-16 to 11-18)

Everything below comes from `results/raw/<experiment>/<job-id>.csv`; `python3 analysis/summarize.py` prints every table (saved as `results/summary.txt`) and `python3 analysis/figures.py` plus `summarize.py --plots results/figures` draw the figures in `results/figures/`. All counts in every run are bit-identical to the serial program (checked by `make test` in the smoke job 31229 and by the GPU jobs), so every speedup below compares identical work.

Allocation limit: QOS `student` gives 64 CPUs per user, so most jobs use 32 physical cores (`--hint=nomultithread`), either on one node or as 2 nodes × 16 cores.

### OpenMP and MPI scaling on one node (jobs 31203, 31204, 31230, 31231)

- **Strong scaling is almost perfect up to 32 cores.** n = 1.2e11 characters, baseline kernel:
  - OpenMP: 165.7 s on 1 thread, 5.21 s on 32 threads, speedup 31.8 (efficiency 0.995).
  - MPI: 166.6 s on 1 rank, 5.23 s on 32 ranks, speedup 31.8 (0.995).
  - 64 threads on the same 32 cores (SMT, 2 threads per core): speedup only 36.4 (efficiency 0.57 per thread). SMT adds ~14%, because the kernel keeps each core's integer units busy.
- **Weak scaling:** efficiency is 0.997 or better at every size from 1 to 32 workers, for OpenMP with close or spread binding and for MPI. Binding makes no difference.
- **Communication:** the only MPI message is one `MPI_Reduce` of the counts at the end, 4.5e-5 to 2.3e-4 s against 5–166 s of compute. Load imbalance (max/min busy time) is 1.000–1.009.
- Why it scales so well: the problem is embarrassingly parallel by construction (independent seeded chunks), compute bound (no memory traffic: ~0.6–1.7 ns per character from registers), and needs one tiny reduction.

### One node vs two nodes, and hybrid MPI + OpenMP (jobs 31338, 31339)

- Same 16 ranks on 1 node or split 8 + 8 over 2 nodes: 10.44 s both. 16 + 16 ranks on 2 nodes: 5.24 s, exactly half. Crossing the node boundary costs nothing measurable for this workload; the reduction takes 0.1–1.9 ms.
- Hybrid on 2 nodes × 16 cores, every split from 32 ranks × 1 thread to 2 ranks × 16 threads: 5.206–5.224 s. 2 × 16 is marginally best (lowest imbalance, 1.004) but all are within 0.4%. With one reduction per run there is nothing for hybrid to save; it matters only when communication is frequent (see stop check below).
- **Problem we hit:** Open MPI's default transport (UCX with shared memory) segfaulted on haskell with 8 or more ranks per node (failed jobs 31232, 31233, logs in `results/logs`). We switched to the plain TCP transport (`OMPI_MCA_pml=ob1`, `btl=self,tcp`, set in `slurm/common.sh`). It does not change the results because we send almost no data.

### Time to solution vs theory (job 31341, first_match, 2 nodes × 16 cores)

- Each run types until the target first appears. The position found is the same for any number of cores, so it can be compared with theory exactly.
- Mean waiting time vs Conway's expected wait (the theory for overlapping patterns):
  - romeo (5 letters, 30 runs): measured 1.436e7 characters, theory 1.435e7 (ratio 1.001)
  - juliet (6, 30 runs): 3.49e8 vs 3.87e8 (0.90)
  - capulet (7, 50 runs): 9.35e9 vs 1.05e10 (0.89)
  - montague (8, 30 runs): 2.70e11 vs 2.82e11 (0.96)
  - wherefore (9, 5 runs): 6.95e12 vs 7.63e12 (0.91)
  - All are within the statistical error (about ±2/√runs, i.e. ±37% for 30 runs and ±90% for 5). The spread (sd/mean 0.6–1.1) matches the exponential-like distribution theory predicts (sd/mean ≈ 1). Figure `first_match_ecdf.png` compares the measured distribution with 1 − exp(−x/E).
- **Headline:** "wherefore" found in all 5 runs on 32 cores, after 102–506 s (2.8e12 to 1.4e13 characters, at 2.78e10 characters/s).
- **Time-to-solution speedup (capulet, same 20 seeds per point, median of per-seed speedups):** 1.55 on 2 cores, 3.93 on 4, 5.90 on 8, 15.3 on 16, 22.4 on 32. Worse than counting because the runs are short (0.2–5 s), ranks agree on "found" only every K chunks, and the work after the hit is wasted (0.1% on 1 core, 2.9% on 32). Figure `time_to_solution.png`.

### Stop-check interval: communication vs wasted work (job 31340)

All ranks must agree regularly whether anyone has found the target. K = chunks per thread between checks (chunk = 2^20 characters). Target montague, 2 nodes × 16 cores, 8 seeds, median time:
- K = 1: sync (blocking `MPI_Allreduce`) 5.40 s with 1.35 s of communication; async (`MPI_Iallreduce` overlapped with the next round) 4.13 s with 0.10 s. Overlap removes 93% of the communication cost and makes the run 24% faster.
- K = 16: sync 4.14 s, async 4.03 s (best overall).
- K = 256: communication is negligible for both, but wasted work grows: async 7.7%, sync 2.9% (async always finishes one extra round). Here sync is better (4.05 vs 4.26 s).
- Lesson: the classic trade-off between communication frequency and wasted work, with a sweet spot at K ≈ 16–64; overlap helps only when communication is frequent. Figure `stop_check_k.png`.

### Optimization: fused kernel, before and after (jobs 31234 = v1, 32267 = v2; old data in `results/old/v1`)

- **Baseline (chunked):** one pass over the text per target, with a multiply-add rolling hash. Cost grows linearly with the number of targets.
- **Optimization 1, fused kernel:** one pass that keeps the last 12 characters in a 64-bit shift register (5 bits per character), compares the window against all targets at once, and uses a small bitmap filter when many targets share a length. Draws exactly the same random characters, so counts are identical.
- **Optimization 2, found from the profile and the cross-check below:** for 1–3 targets the grouped matcher was slower than a plain compare loop, so `fused_scan` now uses the plain loop up to 3 targets.
- Single core, characters/s (v2, job 32267):
  - 1 target: baseline 7.24e8, fused 9.46e8 → 1.31× (v1 was only 1.06×)
  - 2 targets: 4.65e8 → 7.27e8 → 1.56× (v1 1.37×)
  - 4 targets: 2.71e8 → 6.39e8 → 2.36×
  - 8 targets: 1.48e8 → 6.32e8 → 4.28×
  - 16 targets: 7.73e7 → 6.19e8 → 8.0×
  - 8 targets of mixed length (a, oh, the, …): only 1.20×, because every length needs its own comparison group.
- 32 threads, OpenMP: 1 target 2.31e10 → 3.00e10 (1.30×); 16 targets 2.46e9 → 1.95e10 (7.9×).
- Figure `optimization_targets.png` shows the baseline falling as 1/k while the fused kernel stays flat.
- **How optimization 2 was found:** three other code paths (first-match kernel, OpenACC multicore, the GPU kernel's loop run on the CPU) reached ~1.18e9 characters/s per core with the same random stream, against 7.6e8 for our counting kernel. A local microbenchmark showed the general matcher (groups, member lists) cost more than direct comparisons for few targets. A first guess, pointer aliasing between the counters and the generator state, was tested and made no difference.
- **Compiler flags** (single core, 1 target, fused v2): -O2 8.73e8, -O3 9.43e8, -O3 -march=znver4 7.56e8, -O3 -march=znver4 -funroll-loops 9.41e8. -march=znver4 is 20% slower here: AVX-512 does not help a serial dependency chain, and GCC's code for it was worse. Keep plain -O3.

### Profiling and the bottleneck (jobs 31236 = v1 on julia, 32268 = v2 on haskell)

- No hardware counters available to students: `perf_event_paranoid=4` blocks PAPI (and the PAPI module's library failed to load), and Nsight Compute reports `ERR_NVGPUCTRPERM` on the GPUs. We report this as a limitation and use application instrumentation instead: the program times a generate-only pass ("gen") and the full kernels, so matching cost = kernel − gen.
- v2 profile (haskell, ns per character): generating the random character 0.61; baseline with 1/4/16 targets 1.38/3.68/12.91 (matching 0.77/3.08/12.30); fused with 1/4/16 targets 1.05/1.71/1.75 (matching 0.44/1.10/1.14).
- So after the optimization the bottleneck is the random number generator itself: 58% of the time with 1 target, 35% with 16. The next step would be generating several characters per RNG call or a SIMD generator.
- The same profile on julia (v1, job 31236) gave gen = 0.78 ns: julia's cores are ~25% slower than haskell/fortran for this code (also seen in Phase 1: 5.84e8 vs ~7.2e8 characters/s). Compare kernels only within one job.

### Sensitivity: chunk size × OpenMP schedule (job 31235, fused, 32 threads)

- Chunk sizes 2^12 to 2^24 all give 5.52–5.57 s; dynamic,1 and guided are best, static ~1% worse (imbalance 1.016–1.022 vs 1.000).
- Below 2^14 the per-chunk overhead (seeding, scheduling) starts to show (2^12 dynamic: 5.70 s).
- From 2^26 up, there are too few chunks for 32 threads: imbalance 1.07 at 2^28, 1.33 at 2^30, and at 2^32 some threads get no chunk at all. Time rises 14%.
- Default chunk 2^24 sits in the flat region. Figure `sensitivity_chunk.png`.

### GPU: CUDA and OpenACC (jobs 31622, 32119 on fortran = Blackwell; 32120 on haskell = Ada)

- Same algorithm on the GPU: one thread per chunk (grid-stride loop), targets in constant memory, counts reduced with warp shuffles and atomics. Counts are bit-identical to the CPU for all chunk sizes tested.
- Throughput, 1 target: RTX PRO 6000 Blackwell 1.15e12 characters/s, RTX 6000 Ada 8.6e11. That is about 38× and 29× the best 32-core CPU result (3.0e10), or ~1200× one core.
- Block size (64–1024 threads): within 3%; 512 is best on both cards.
- Chunk size (1K–4M): flat on Blackwell; on Ada 4M chunks drop to 6.6e11 (too few chunks to fill the GPU).
- Number of targets: 1 / 4 / 16 targets give 1.15e12 / 7.0e11 / 2.8e11 on Blackwell. The GPU version uses the simple compare loop without the bitmap filter, so it slows down with many targets (future work).
- **Headline:** one ~6-minute GPU run (3.5e14 characters) finds "wherefore" ~50 times (theory 45) and "friar john" (10 characters including the space) twice (theory 1.7). Ada: 37 and 1 (theory 33.5 and 1.2).
- **OpenACC:** same loop with `#pragma acc parallel loop` and a reduction, built with nvhpc 25.7. On the GPU it reaches 7.0e11 (Blackwell) and 5.5e11 (Ada): 0.61–0.64× of hand-written CUDA with far less code. Built for multicore CPU (`-acc=multicore`, 4 cores) it beat our GCC OpenMP build on the same cores (4.7e9 vs 2.9e9), which is what pointed us to optimization 2.
- **Problems we hit:** `nvcc` was not on PATH after `module load hpc-course-cuda/12.8.1` (job 31342); the job now searches for it and falls back to the nvcc in nvhpc. On haskell it found the system CUDA 12.4, which cannot compile for Blackwell (compute_120) (job 31621); the job now compiles only for the GPU it runs on.

### Projections (for the EuroHPC plan)

Expected time to find a target of length L = Conway's E(L) ≈ 27^L divided by throughput. With measured rates (Blackwell 1.15e12 /s, 32 CPU cores 3.0e10 /s):
- 9 letters ("wherefore", E = 7.6e12): 6.6 s on one Blackwell GPU, 4.2 min on 32 cores.
- 10 characters (E ≈ 2.1e14): 3 min on one GPU.
- 11 characters (E ≈ 5.6e15): 1.4 h on one GPU, ~5 min on 16 GPUs.
- 12 characters (E ≈ 1.5e17): 36 h on one GPU, ~2.3 h on 16 GPUs, ~9 min on 256 GPUs.
- Every extra letter costs 27× more; scaling is near-linear, so the plan is a question of GPU-hours: E(L) / 1.15e12 / 3600 GPU-hours per expected hit (12 letters ≈ 36 GPU-hours, 13 ≈ 970, 14 ≈ 26,000).
