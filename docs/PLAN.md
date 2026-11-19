# Infinite Monkeys on SciTech — Project Plan

## 1. What we are building (the professor's version)

Many "monkeys" type uniformly random characters from a 27-symbol alphabet (a–z plus space). We count how often short target strings from Romeo and Juliet appear, and how long it takes until a target first appears. All work is spread with MPI and OpenMP, launched through Slurm, and checked against exact theory.

The professor's conditions, which we treat as the grading rubric:
- the focus is parallel performance, not text generation;
- MPI and/or OpenMP, with speedup and efficiency measured across cores and nodes;
- results compared against theoretical probabilities;
- short targets only;
- proper measurements, scalability analysis and reproducible experiments.

One-line thesis for the report: *the kernel is embarrassingly parallel, so the interesting HPC questions are how fast a single core can type, where the remaining overheads come from (RNG, matching, termination signalling, load imbalance), and how far more hardware pushes the longest target we can find.*

## 2. Why it is a good HPC problem (numbers to quote)

- Expected occurrences of a length-L target in n characters: (n − L + 1) / 27^L.
- Expected characters until the first occurrence (Conway): sum of 27^j over every j where the length-j prefix equals the length-j suffix. For "the", 27^3 ≈ 2.0e4.
- 27^L grows about 27× per extra character: L=5 → 1.4e7, L=7 → 1.0e10, L=9 → 7.6e12, L=10 → 2.1e14, L=11 → 5.6e15.
- Measured on the cluster (EPYC 9745, job 30127): 5.84e8 chars/s on one core. With perfect scaling:
  - "wherefore" (9 chars) takes ~51 s on 256 cores (one node);
  - a 10-char target takes ~23 min on one node, or ~11 min on two;
  - an 11-char target takes ~10 h on one node.
- Cluster: 2 usable nodes (haskell, julia; fortran planned), each with 256 cores and 512 hardware threads. Details and measured findings are in docs/FINDINGS.md.
- Headline experiment: **"how many characters can our hardware afford?"** Every 27× speedup (better kernel, more nodes, GPUs) buys one more character of Shakespeare. That gives a scaling story the audience understands immediately.

## 3. Current state (commit d79ea7f, by Neo)

Done:
- `src/rng.h`: xoshiro256** with jump-ahead for independent streams.
- `src/monkey.h`: target parsing, unbiased 0..26 draw (Lemire), rolling base-27 window matcher, expected count and Conway wait.
- `src/serial.c`: CSV output.
- `analysis/check_theory.py`: z-score test. It passes locally.

Empty skeletons: `omp.c`, `mpi.c`, `hybrid.c`, `first_match.c`, all `.sbatch`, `run_all.sh`, `theory.py`, `plot_scaling.py`, `README.md`.

Things to fix early:
- `monkey_count` scans each 8 KB block once per target, which explains 4 targets being ~3.3× slower than 1. Keep this version as the "before" in the optimization section and don't fix it yet.
- Decomposition is not decided yet; see section 4.1.
- The Makefile only builds `serial`.
- The brief recommends `src/ env/ slurm/ data/ results/ docs/` plus `README.md` and `REPRODUCE.md`. Move `scripts/slurm/` to `slurm/` and add `env/` and `data/` now, before more files reference the old paths.

## 4. Design decisions (agree on these before writing more code)

### 4.1 Decompose into fixed chunks, not per worker (bit-reproducible results)

Split the total n into fixed chunks of C characters (e.g. C = 2^28). Chunk i always uses RNG stream i, and workers take chunks. The counts are then **bit-identical for any number of ranks or threads**. Benefits:
- the correctness test becomes exact: `serial == omp == mpi == hybrid == cuda` on the same (n, seed);
- the reproducibility section of the report is trivial.

Costs:
- Windows spanning a chunk boundary are not counted. The expected count becomes the sum over chunks of (C − L + 1) / 27^L. Put that in one function used by C and Python.
- `rng_seed` jumps stream_id times, so seeding chunk 10^6 costs 10^6 jumps. Instead, seed chunk i from `splitmix64(seed ^ i)`, or jump incrementally inside each worker. Document the choice and its independence argument.

C becomes a sensitivity parameter: small chunks balance load better but cost more scheduling overhead.

### 4.2 Three programs, one kernel header

- `count` mode (serial/omp/mpi/hybrid): type n characters and count every target. Used for scaling, theory checks and optimization.
- `first_match`: type until any worker finds the target, then stop everyone. This is the "time-to-solution" experiment and the only part with real communication. Termination options:
  - MPI: each rank checks `MPI_Iallreduce` or `MPI_Test` on a flag every K chunks.
  - OpenMP: a shared atomic flag checked every chunk.
  - K is a sensitivity parameter. A small K means more communication; a large K means wasted work after the hit. We measure both sides.
- Stretch, "how far we get": the longest substring of the whole play typed by the monkeys. A suffix automaton of the play (~150k chars) is scanned with the random stream and gives the longest match ending at each position. Theory predicts a longest match of about log_27(n × play_length) characters, roughly 12–13 for n = 1e13.
  - This kernel is **memory-latency bound** (random walks over a multi-MB automaton), while the count kernel is **compute bound**. Having one of each is exactly what the profiling section needs.
  - Only attempt it once the core work is done (gate G4).

### 4.3 Output contract

Every binary prints the same CSV:
`version,ranks,threads,n,seed,target,len,count,expected,time_s,chars_per_s`
Add `node_list`, `chunk`, `git_sha` and a `phase_times` breakdown (generation, matching, reduction). Every Slurm job appends to `results/raw/<experiment>.csv` and copies its `sacct` line to `results/logs/`.

## 5. Phases and gates

Gates, not dates: we don't start a phase until the previous gate passes. Map these to the Blackboard milestones once dates are known.

### Phase 0 — Cluster setup (gate G0)
- On a compute node, through `srun`, record:
  - `sinfo -o "%P %D %c %m %G"`
  - `lscpu`
  - `module avail`
  - core count per node and number of CPU nodes
- Save the output to `env/cluster.txt`. This sets the maximum of our scaling range.
- `env/modules.sh` loads OpenMPI/4.1.5-GCC-12.3.0. Check `which mpicc` after loading, because a system `/usr/bin/mpicc` has shadowed the module before. Use `module load` + `mpicc --showme` and log the result.
- Python venv with numpy/pandas/matplotlib, frozen to `env/requirements.txt`.
- **G0:** `slurm/serial.sbatch` runs on a compute node and writes a CSV into `results/raw/`.

### Phase 1 — Baseline (milestone 2, gate G1)
- Apply the chunk decomposition (4.1) to `monkey.h`, keeping the original single-stream kernel as `monkey_count_v0` for the optimization comparison.
- Baseline on 1 core of a cluster node, 5 repetitions:
  - targets of length 1–7;
  - 1 target vs 4 vs 16 targets.
- `check_theory.py` on the cluster.
- **G1:** baseline CSV and theory check committed from cluster runs, plus a short note on single-core chars/s.

### Phase 2 — OpenMP (milestone 3, gate G2)
- `omp.c`: `#pragma omp parallel for schedule(dynamic)` over chunks, with per-thread counts reduced at the end.
- Strong scaling: fixed n, threads 1, 2, 4, 8, 16, … up to all cores of one node.
- Weak scaling: n per thread fixed.
- Pinning: `OMP_PROC_BIND=close/spread`, `OMP_PLACES=cores`. Record whether SMT is on.
- **G2:** identical counts to serial for the same (n, seed). Speedup and efficiency plotted. Any efficiency loss is explained (turbo frequency drop, memory, scheduling).

### Phase 3 — MPI and hybrid (milestone 4, gate G3)
- `mpi.c`: static block or cyclic chunk assignment, `MPI_Reduce` of counts, `MPI_Wtime` around compute and around the reduction separately.
- `hybrid.c`: MPI ranks × OpenMP threads.
- Experiments:
  - 1 → 2 → 4 nodes (or all we can get);
  - ranks per node × threads per rank sweep on a fixed core count (e.g. 64×1, 16×4, 4×16, 2×32);
  - one node vs two nodes at equal core count.
- **G3:** identical counts to serial. Multi-node runs done. Communication time shown as a fraction of total time.

### Phase 4 — First match and time-to-solution (gate G4)
- `first_match.c` (MPI + OpenMP) with termination check every K chunks.
- For each target length (5, 6, 7, then 8 and 9 on many cores), run ≥ 30 trials with different seeds.
  - Compare the mean and the distribution with Conway's expected wait. The distribution should be roughly geometric/exponential, so plot the ECDF against the theoretical CDF.
- Time-to-solution against core count. Expected speed-up is linear, but variance is huge, so report the median and an interquartile band. Statistics, not a single run, is the point.
- K sensitivity: termination latency and wasted characters against communication overhead.
- **G4:** "wherefore" (or another 9-char target) found on the cluster, with logs.

### Phase 5 — Profiling and optimization (milestone 5, gate G5)

Profiling, in this order:
- `perf stat` on the serial kernel: IPC, instructions per char, branch misses (the rejection-sampling branch), cache misses. This shows whether the kernel is compute bound.
- `perf record` / `perf report`: share of time in `rng_next` + `monkey_draw` vs the matching loop.
- Application timers per phase (already in the CSV).
- MPI time from `MPI_Wtime` around the reduction and termination checks. Optional: mpiP or the OpenMPI PMPI interface, if available.
- `sacct --format=JobID,Elapsed,TotalCPU,MaxRSS,AveCPUFreq` for CPU utilization and memory.
- If LIKWID is available: `likwid-perfctr -g FLOPS_DP/MEM` to confirm low memory bandwidth (and, for the stretch automaton kernel, high bandwidth or latency).

Optimization candidates. Each one is measured before and after on 1 core and on a full node:
1. **Fused single pass over all targets.** Keep one window and compare against all targets, instead of scanning once per target. Expected: 4 targets cost about the same as 1, so ~3× on the 4-target workload.
2. **5-bit shift register instead of ×27 rolling hash.** `w = ((w << 5) | c) & mask`. Each char needs 5 bits and 12 chars fit in 60 bits. Replaces a 3-cycle multiply chain with 1-cycle shift/or and lets a hash set test many targets at once.
3. **Cheaper random characters.** Get more than 2 chars per 64-bit RNG call. For example, take 5-bit chunks with rejection of values ≥ 27 (about 11 chars per call), or use a multiply-chain that stays exactly uniform. The bias must stay zero, and the theory check proves it.
4. **SIMD across monkeys.** Run 4/8 independent xoshiro streams in AVX2/AVX-512 lanes (`-O3 -march=native`, check vectorization with `-fopt-info-vec`). This is the largest potential gain because the rolling window is a serial dependency within one stream but independent across streams.
5. Compiler flags sweep: `-O2` vs `-O3 -march=native` vs `-flto`. Also GCC vs any other compiler available. This is cheap and belongs in the sensitivity section.

Pick 1 (+2) as "the" required optimization. 3–5 are bonuses.

- **G5:** before/after numbers, with perf evidence showing *why* the time changed (instructions/char, IPC). Theory check still passes after every optimization.

### Phase 6 — Additional technology: CUDA (gate G6)
- `cuda/monkey.cu`: one thread = one monkey stream, Philox (cuRAND device API) or xoshiro per thread, the same chunk → stream mapping as the CPU version where feasible, and counts reduced with warp shuffles + atomics.
- Runs on the GPU partition, 1 GPU, short jobs (≤ 10 min). Compare Ada vs Blackwell if both are available. Profile with Nsight Systems/Compute (achieved occupancy, integer throughput, registers).
- Report chars/s per GPU vs per CPU node and characters gained: if the GPU is 27× faster than a node, it buys one more letter.
- Optional second experiment: OpenACC port of the same kernel with `nvc` (if the NVHPC module exists). A negative result is fine.
- **G6:** GPU result plus a technology-matrix row saying whether it stays in the final application.

### Phase 7 — Write-up (final)
- `analysis/plot_scaling.py`: speedup, efficiency, weak scaling, hybrid sweep, before/after, ECDF vs theory, chars/s per technology.
- `REPRODUCE.md`: exact commands for the main figures from a clean clone, including git tag, modules, flags, seeds and sbatch lines.
- Report (5–7 pages), EuroHPC plan (1–2 pages) and slides (5) in `docs/`.

## 6. Experiment matrix (what ends up in the report)

- **Baseline:** serial, 1 core, targets L = 1..7, 1/4/16 targets, 5 repetitions each.
- **Strong scaling:** n = 1e11 (pick so 1 core takes ~5 min), cores 1 → max single node (OpenMP) → 2/4 nodes (MPI, hybrid).
- **Weak scaling:** 2e9 chars per core, cores 1 → max.
- **Hybrid sweep:** fixed total cores, ranks × threads combinations.
- **Sensitivity:** chunk size C (2^16 … 2^32), number of targets, target length, termination interval K, compiler flags.
- **Time-to-solution:** first match for L = 5..9, ≥ 30 seeds each, against core count.
- **Optimization:** v0 vs fused vs shift-register vs SIMD, 1 core and full node.
- **Technology:** CPU node vs GPU (CUDA), optional OpenACC.
- **Every run:** at least 3 repetitions. Report the median and min–max, with the job ID logged.

Rules: never benchmark on the login node, use exclusive nodes (`--exclusive`) for scaling runs, record CPU frequency/turbo state, and run every scaling point in the same job when possible so the hardware state stays consistent.

## 7. Repository layout (target)

```
src/        serial.c omp.c mpi.c hybrid.c first_match.c monkey.h rng.h
cuda/       monkey.cu
env/        modules.sh, cluster.txt, requirements.txt, compiler versions
slurm/      one .sbatch per experiment, each takes parameters via env vars
data/       get_play.sh (Project Gutenberg Romeo and Juliet, public domain), targets.txt
results/    raw/*.csv, logs/, figures/
analysis/   theory.py check_theory.py plot_scaling.py
docs/       PLAN.md, report, eurohpc_plan, slides
README.md   REPRODUCE.md   Makefile
```

## 8. Report and EuroHPC notes

- The report must interpret every graph. For example: "efficiency drops to 0.85 at 64 threads because all-core turbo is 15% lower than single-core turbo (AveCPUFreq evidence)", not "it scales well".
- Technology matrix rows:
  - Slurm
  - OpenMP
  - MPI
  - Hybrid
  - SIMD/vectorization
  - CUDA
  - OpenACC (if tried)
  - perf/LIKWID
  - Nsight
- EuroHPC plan:
  - The application is embarrassingly parallel with one tiny reduction, so a GPU-accelerated system (e.g. LUMI-G, Leonardo Booster, MareNostrum 5 ACC) is the right target. Justify it from our measured GPU vs CPU-node chars/s.
  - Size the request by target length. Each extra character costs 27× the node-hours, so derive node-hours for L = 10, 11 and 12 from our measured throughput.
  - Main scaling challenges at large scale: termination latency at 10^4+ ranks, stragglers, and job-length limits (checkpointing the counters).
- Slides: problem, then "every 27× buys one letter", then scaling plots, then bottleneck + optimization, then lessons + EuroHPC.

## 9. Suggested split (4 people, adjust to the group)

- **Kernel + optimization owner:** monkey.h, chunk decomposition, fused/shift/SIMD versions, perf profiling.
- **Parallel owner:** omp.c, mpi.c, hybrid.c, first_match.c termination logic.
- **Experiments + reproducibility owner:** env/, slurm/, run_all.sh, results pipeline, REPRODUCE.md, sacct logging.
- **GPU + analysis owner:** CUDA port, Nsight, theory.py, plot_scaling.py, figures.

Everyone writes the report section for their own part. One person edits the whole report for one voice.

## 10. Risks

- **Too few nodes or queue waits:** keep jobs short and run sweeps inside one allocation.
- **Statistical noise in first-match timing:** many seeds and medians, never single runs.
- **GPU contention:** short jobs only, done after the CPU story is complete.
- **Scope creep (automaton stretch, OpenACC):** only after G5.
- **Results that silently go wrong after an optimization:** the exact-count equality test and the theory z-test run in CI style (`make test`) before every commit.
