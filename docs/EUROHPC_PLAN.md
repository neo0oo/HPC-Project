# EuroHPC scale-up plan

Group 2, Infinite Monkeys on SciTech. Every measured number below comes from [FINDINGS.md](FINDINGS.md), with job IDs there.

## 1. Why a larger system would help

The cost of the problem is set by the target length: a target of L characters first appears after about 27^L random characters, so every extra character costs 27× more work. On SciTech we are limited to 32 cores or 1 GPU per user, which gave us:

| Platform (measured) | Characters/s | Longest target found |
|---|---|---|
| 32 CPU cores, fused kernel | 3.0e10 | 9 ("wherefore", 102–506 s) |
| 1 RTX 6000 Ada, 6-minute run | 7.0e11 | 10 ("friar john", once) |
| 1 RTX PRO 6000 Blackwell, 6-minute run | 9.2e11 | 10 ("friar john", twice) |

One more character needs 27× the hardware or 27× the time; two more need 729×. Our scaling is near perfect (efficiency 0.995 at 32 cores, no cost for crossing nodes), so extra hardware converts almost directly into longer targets. A larger system would also let us test the two things we could not test here: scaling beyond 2 nodes, and the stop signal of the time-to-solution program with hundreds of ranks.

## 2. Target system

A GPU-accelerated EuroHPC system with NVIDIA GPUs, for example the **Leonardo Booster** partition (about 3,456 nodes, each with 4 NVIDIA A100 GPUs, 32 CPU cores and 512 GB RAM) or MareNostrum 5 ACC.

- **Why GPU:** one GPU does the work of about 30 of our 32-core allocations (9.2e11 vs 3.0e10 characters/s). If the CPU result scaled perfectly to a full 256-core node (not measured, the QOS stops us at 32 cores), one GPU would still equal about 4 full CPU nodes.
- **Why NVIDIA:** our GPU version is CUDA. LUMI-G uses AMD GPUs and would need a HIP port first; our OpenACC version reached only 0.61–0.64× of CUDA.
- **Access route:** a EuroHPC Benchmark or Development access call, which are meant for scaling tests of this size.

## 3. Requirements

- **GPUs:** 4 per node, up to 256 nodes (1,024 GPUs) for the largest runs.
- **CPUs:** one core per GPU to drive it; the CPU does no counting.
- **Memory:** very small. The program never stores the text: each GPU thread keeps a 256-bit generator state, a 64-bit window and its counters. Under 1 GB per GPU and a few GB of host memory per node.
- **Storage and I/O:** CSV results and logs only, under 1 GB in total. No parallel file system load.
- **Software:** GCC 12 or newer, CUDA 12 or newer, an MPI library started through Slurm (we used OpenMPI 4.1.5), Python 3.11 with matplotlib for the analysis. Optional: NVHPC for the OpenACC comparison, Nsight Systems/Compute for profiling.

## 4. Planned experiments and resource request

**Planning rate.** We have no A100 measurement. We assume 4e11 characters/s per A100, about half of our RTX 6000 Ada result, so 1.6e12 per node. This is the main uncertainty and the first thing the validation runs will measure. We use the 6-minute "headline" rates above, not the 1-second sweep rates (1.15e12 on Blackwell), because the long runs were 20% slower.

Expected cost of one hit at 1.6e12 characters/s per node: 12 characters (27^12 = 1.5e17) takes 26 node-hours; 13 characters (4.1e18) takes 704 node-hours; 14 characters would take 19,000 node-hours and is out of scope.

| # | Experiment | Size | Runs | Node-hours |
|---|---|---|---|---|
| A | Port to MPI + CUDA, exact-count checks against the CPU version | 4 nodes × 0.5 h | 10 | 20 |
| B | Strong scaling, 2e16 characters, 1 → 256 nodes (9 sizes) | 3.5 node-h per run | 27 | 95 |
| C | Weak scaling, 10 minutes per run, 1 → 256 nodes (9 sizes) | 511 nodes × 1/6 h | 27 | 256 |
| D | Time to solution, 12-character target, 30 seeds, on 64 nodes | 26 node-h expected per run (~25 min) | 30 | 780 |
| E | Time to solution, 13-character target, 5 seeds, on 256 nodes | 704 node-h expected per run (~2.7 h) | 5 | 3,515 |
| F | Stop-check interval and sync vs async at scale, 64 nodes | 64 nodes × 0.25 h | 20 | 320 |
| | **Total** | | **119** | **4,986** |

Waiting times are close to exponentially distributed (we measured sd/mean of 0.6–1.1), so single runs in D and E can take three times the expected value. With 20% contingency for that and for failed jobs:

```
4,986 node-hours × 1.2 ≈ 6,000 node-hours  (= 24,000 GPU-hours)
```

Run duration: from 1 minute (256-node strong-scaling point) to about 8 hours (a slow 13-character run), all inside a normal 24-hour job limit.

## 5. Expected scaling challenges

- **The GPU version is single-GPU today.** It needs an MPI layer with one rank per GPU. The work split already exists: chunk i is seeded only from (seed, i), so ranks just take different chunk ranges and counts stay bit-identical.
- **Stop signal at scale.** With 32 cores, checking every chunk with a blocking `MPI_Allreduce` cost 1.35 s of a 5.4 s run; the non-blocking version removed 93% of that. With 1,024 ranks the reduction is slower and each GPU does far more work between checks, so the check interval must be retuned (experiment F). Wasted work after the hit already grew from 0.1% on 1 core to 2.9% on 32.
- **Variance.** Time to solution is random, so speedup must come from many seeds and medians, which is why D and E dominate the request.
- **Many targets on the GPU.** The GPU kernel slows from 1.15e12 (1 target) to 2.8e11 (16 targets) because it lacks the bitmap filter of the CPU kernel. We will run one target at a time or port the filter.
- **Counter width.** Positions are 64-bit (up to 1.8e19). A 13-character run fits; 14 characters (1.1e20) would need wider counters.
- **Job limits.** A long run must be restartable. A checkpoint is only the next chunk index and the counters.

## 6. Support we would need

- Help choosing the MPI + CUDA launch setup (one rank per GPU, binding) on the target system.
- Access to GPU and CPU hardware counters. On SciTech both PAPI and Nsight Compute were blocked by permissions, so our bottleneck analysis used application timers only.
- Advice on queue policy for many short jobs (experiments B, C and F) versus a few long ones (E).
