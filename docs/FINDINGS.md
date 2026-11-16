# Findings log

Running record of measured results, with job IDs, for the report.

## Phase 0: cluster (probe job 30126, 2026-11-15)

- **Nodes:** 2 usable today: haskell (2× RTX 6000 Ada) and julia (1× RTX PRO 6000 Blackwell). A third, fortran (2× Blackwell), is planned. The nodes are shared between the cpu, gpu and interactive partitions. Time limits: cpu 7 days, gpu 3 days, interactive 8 hours.
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
