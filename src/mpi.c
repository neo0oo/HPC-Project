/*
 * mpi.c - MPI version (pure MPI), and the MPI + OpenMP hybrid when compiled
 * with OpenMP (the Makefile builds both from this file: bin/mpi, bin/hybrid).
 *
 *     srun -n 32 ./mpi -n 1.2e11 -t romeo
 *     OMP_NUM_THREADS=8 OMP_PLACES=cores srun -n 4 -c 8 ./hybrid -n 1.2e11
 *
 * Decomposition: rank r owns the contiguous block of chunks
 * [r*nchunks/P, (r+1)*nchunks/P). In the hybrid, the rank's threads share its
 * block dynamically (OMP_SCHEDULE, default dynamic,1). Because chunk i is
 * always typed by generator (seed, i), the counts are bit-identical to
 * ./serial for any P x T.
 *
 * Timing (all from MPI_Wtime, after a barrier):
 *   time_s   total wall time on rank 0, compute + final reduction
 *   comm_s   time of the final MPI_Reduce of the counters (the only
 *            communication), max over ranks. A barrier before it absorbs
 *            the wait for the slowest rank, so comm_s is pure communication;
 *            the waiting shows up as busy_max - busy_min instead.
 *   busy_*   shortest / longest compute time of any rank x thread
 */
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#define VERSION "hybrid"
#else
#define VERSION "mpi"
#endif

#include "cli.h"

int main(int argc, char **argv)
{
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Only rank 0 prints errors / usage; everyone parses the same args. */
    monkey_opts_t o;
    monkey_targets_t targets;
    if (rank != 0)
        freopen("/dev/null", "w", stderr);
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0) {
        MPI_Finalize();
        return rc;
    }
    if (o.kernel == KERNEL_V0) {
        if (rank == 0)
            fprintf(stderr, "error: v0 is serial only\n");
        MPI_Finalize();
        return 2;
    }
    monkey_work_t work;
    work_init(&work, o.kernel, &targets);

    const uint64_t nchunks = monkey_num_chunks(o.n, o.chunk);
    const uint64_t first = nchunks * (uint64_t)rank / (uint64_t)size;
    const uint64_t last = nchunks * (uint64_t)(rank + 1) / (uint64_t)size;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
    if (getenv("OMP_SCHEDULE") == NULL)
        omp_set_schedule(omp_sched_dynamic, 1);
#endif

    uint64_t local[MONKEY_MAX_TARGETS] = {0};
    double busy_min = 1e300, busy_max = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

#ifdef _OPENMP
#pragma omp parallel
    {
        uint64_t mine[MONKEY_MAX_TARGETS] = {0};
        double b0 = omp_get_wtime();
#pragma omp for schedule(runtime) nowait
        for (uint64_t i = first; i < last; i++)
            work_chunk(&work, o.n, o.chunk, o.seed, i, mine);
        double busy = omp_get_wtime() - b0;
#pragma omp critical
        {
            for (int k = 0; k < targets.n; k++)
                local[k] += mine[k];
            if (busy < busy_min)
                busy_min = busy;
            if (busy > busy_max)
                busy_max = busy;
        }
    }
#else
    for (uint64_t i = first; i < last; i++)
        work_chunk(&work, o.n, o.chunk, o.seed, i, local);
    busy_min = busy_max = MPI_Wtime() - t0;
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    double c0 = MPI_Wtime();
    uint64_t counts[MONKEY_MAX_TARGETS] = {0};
    MPI_Reduce(local, counts, targets.n, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    double comm = MPI_Wtime() - c0;
    double elapsed = MPI_Wtime() - t0;

    double gmin, gmax, gcomm;
    MPI_Reduce(&busy_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&busy_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm, &gcomm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        monkey_print_csv(VERSION, size, threads, &o, &targets, counts,
                         elapsed, gmin, gmax, gcomm);
    MPI_Finalize();
    return 0;
}
