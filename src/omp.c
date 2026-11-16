/*
 * omp.c - OpenMP version: threads share the chunks of one run.
 *
 * Usage: same options as serial (see cli.h). Thread count, scheduling and
 * pinning come from the standard OpenMP environment:
 *     OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close ./omp -n 1e11
 *     OMP_SCHEDULE=static ./omp ...      (default: dynamic,1)
 *
 * Chunk i is always typed by generator (seed, i), so the counts are
 * bit-identical to ./serial for the same (n, chunk, seed) whatever the number
 * of threads or the schedule.
 *
 * Each thread accumulates into private counters (no sharing in the hot loop)
 * and records how long it was busy; busy_min/busy_max in the CSV expose load
 * imbalance (e.g. 1000 chunks over 256 threads -> some threads get 4, some 3).
 */
#include <omp.h>

#include "cli.h"

int main(int argc, char **argv)
{
    monkey_opts_t o;
    monkey_targets_t targets;
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0)
        return rc;
    if (o.kernel != KERNEL_CHUNKED) {
        fprintf(stderr, "error: omp supports only the chunked kernel\n");
        return 2;
    }

    /* Default schedule dynamic,1 unless OMP_SCHEDULE says otherwise. */
    if (getenv("OMP_SCHEDULE") == NULL)
        omp_set_schedule(omp_sched_dynamic, 1);

    const uint64_t nchunks = monkey_num_chunks(o.n, o.chunk);
    const int nthreads = omp_get_max_threads();
    uint64_t counts[MONKEY_MAX_TARGETS] = {0};
    double busy_min = 1e300, busy_max = 0.0;

    double t0 = omp_get_wtime();
#pragma omp parallel
    {
        uint64_t local[MONKEY_MAX_TARGETS] = {0};
        double b0 = omp_get_wtime();

#pragma omp for schedule(runtime) nowait
        for (uint64_t i = 0; i < nchunks; i++)
            monkey_count_chunk(o.n, o.chunk, o.seed, i, &targets, local);

        double busy = omp_get_wtime() - b0;
#pragma omp critical
        {
            for (int k = 0; k < targets.n; k++)
                counts[k] += local[k];
            if (busy < busy_min)
                busy_min = busy;
            if (busy > busy_max)
                busy_max = busy;
        }
    }
    double elapsed = omp_get_wtime() - t0;

    monkey_print_csv("omp", 1, nthreads, &o, &targets, counts, elapsed,
                     busy_min, busy_max);
    return 0;
}
