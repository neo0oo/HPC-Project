# Infinite Monkeys on SciTech

Many simulated "monkeys" type uniformly random characters (a–z and space, 27 symbols). We count how often short Romeo and Juliet targets appear, measure how long the first occurrence takes, and compare everything with exact theory. The work is spread over cores and nodes with OpenMP and MPI through Slurm, and we measure speedup, efficiency, bottlenecks and optimizations.

Every extra character in a target is 27× rarer, so every 27× speedup buys one more letter of Shakespeare.

The full plan is in [docs/PLAN.md](docs/PLAN.md), and the commands for every result are in [REPRODUCE.md](REPRODUCE.md).

## Quick start

```bash
make                 # builds bin/serial
make test            # theory z-test + exact reproducibility checks
bin/serial -n 1e9 -t romeo,juliet -H
python3 analysis/theory.py --rate 3e8 --cores 64 wherefore
```

Options (shared by every version): `-n` characters (`1e9`, `2^30`), `-t` comma-separated targets (`_` = space), `-s` seed, `-c` chunk size (default `2^24`), `-k chunked|v0`, `-H` CSV header.

## How the work is decomposed

The n characters are split into fixed chunks. Chunk i is typed by its own generator, seeded only from (seed, i). Counts are therefore bit-identical for any number of threads, ranks or nodes, and every parallel version is checked against the serial one exactly (`analysis/check_repro.py --compare`). Windows that cross a chunk boundary are not counted, and the theory accounts for that (`expected_count_chunked`).

## Layout

```
src/        monkey.h (kernel + theory), rng.h (xoshiro256**), cli.h (options/CSV), serial.c, ...
env/        modules.sh (toolchain), cluster/ (probe output per node type)
slurm/      one .sbatch per experiment, common.sh, run_all.sh
data/       targets/*.txt
results/    raw/<experiment>/<job-id>.csv, logs/ (job output + provenance .env)
analysis/   theory.py, check_theory.py, check_repro.py, plotting
docs/       PLAN.md, report, EuroHPC plan, slides
```
