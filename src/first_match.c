/*
 * first_match.c - time-to-solution: type until the target first appears.
 * MPI + OpenMP (works with 1 rank and/or 1 thread).
 *
 *     OMP_NUM_THREADS=16 srun -n 2 -c 16 ./first_match -t capulet -s 7 -n 1e15
 *         -K 16      chunks per thread between stop checks
 *         -m sync    blocking check (MPI_Allreduce) after every round
 *         -m async   overlapped check: post MPI_Iallreduce, compute the next
 *                    round, then wait (communication hidden behind work, at
 *                    the price of one extra round of work after a hit)
 *         -n         give up after this many characters
 *         -k fused   kernel (chunked or fused)
 *
 * The global text is the sequence of chunks 0, 1, 2, ... (chunk i typed by
 * generator (seed, i)). Work proceeds in ROUNDS of W = ranks*threads*K
 * consecutive chunks; rank p takes the p-th slice of each round and its
 * threads share it. After a round every chunk below the round's end has been
 * typed, so the smallest hit found is the true first occurrence in the
 * global text. Hence first_pos is the SAME for any ranks x threads x K
 * (checked by analysis/check_repro.py); only the wasted work and the time
 * change.
 *
 * first_pos is compared with Conway's expected waiting time (theory.py);
 * across many seeds it should be ~geometric with that mean.
 *
 * Output CSV (FM_HEADER): one row per run.
 */
#include <mpi.h>
#include <omp.h>

#include "cli.h"

#define FM_HEADER                                                         \
    "version,kernel,ranks,threads,seed,target,len,chunk,check,mode,"      \
    "first_pos,expected_wait,chars_typed,wasted_frac,rounds,time_s,"      \
    "comm_s,chars_per_s,host,git"

#define NONE UINT64_MAX

/* Process one round: chunks [lo, hi) of this rank. Returns this rank's
 * smallest global hit position (NONE if none); adds typed characters. */
static uint64_t do_round(const monkey_work_t *w, const monkey_opts_t *o,
                         uint64_t lo, uint64_t hi, uint64_t *typed)
{
    uint64_t best = NONE, tsum = 0;
#pragma omp parallel for schedule(runtime) reduction(min : best) \
    reduction(+ : tsum)
    for (uint64_t i = lo; i < hi; i++) {
        uint64_t len = monkey_chunk_len(o->n, o->chunk, i), t = 0;
        uint64_t pos = work_chunk_first(w, len, o->seed, i, &t);
        tsum += t;
        if (pos && i * o->chunk + pos < best)
            best = i * o->chunk + pos;
    }
    *typed += tsum;
    return best;
}

int main(int argc, char **argv)
{
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank != 0)
        freopen("/dev/null", "w", stderr);

    monkey_opts_t o;
    monkey_targets_t targets;
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0) {
        MPI_Finalize();
        return rc;
    }
    if (targets.n != 1 || o.kernel == KERNEL_V0) {
        if (rank == 0)
            fprintf(stderr, "error: first_match needs exactly one target "
                            "and -k chunked|fused\n");
        MPI_Finalize();
        return 2;
    }
    if (getenv("OMP_SCHEDULE") == NULL)
        omp_set_schedule(omp_sched_dynamic, 1);

    monkey_work_t work;
    work_init(&work, o.kernel, &targets);

    const int threads = omp_get_max_threads();
    const uint64_t slice = (uint64_t)threads * o.check;   /* per rank  */
    const uint64_t W = slice * (uint64_t)size;            /* per round */
    const uint64_t nchunks = monkey_num_chunks(o.n, o.chunk);

    uint64_t typed = 0, found = NONE, rounds = 0;
    double comm = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (!o.async) {
        for (uint64_t base = 0; base < nchunks && found == NONE; base += W) {
            uint64_t lo = base + (uint64_t)rank * slice;
            uint64_t hi = lo + slice;
            if (lo > nchunks) lo = nchunks;
            if (hi > nchunks) hi = nchunks;
            uint64_t mine = do_round(&work, &o, lo, hi, &typed);
            double c0 = MPI_Wtime();
            MPI_Allreduce(&mine, &found, 1, MPI_UINT64_T, MPI_MIN,
                          MPI_COMM_WORLD);
            comm += MPI_Wtime() - c0;
            rounds++;
        }
    } else {
        /* Round r's result travels while round r+1 is computed. */
        uint64_t sendbuf = NONE, recvbuf = NONE;
        MPI_Request req = MPI_REQUEST_NULL;
        int pending = 0;
        for (uint64_t base = 0; base < nchunks; base += W) {
            uint64_t lo = base + (uint64_t)rank * slice;
            uint64_t hi = lo + slice;
            if (lo > nchunks) lo = nchunks;
            if (hi > nchunks) hi = nchunks;
            uint64_t mine = do_round(&work, &o, lo, hi, &typed);
            rounds++;
            if (pending) {
                double c0 = MPI_Wtime();
                MPI_Wait(&req, MPI_STATUS_IGNORE);
                comm += MPI_Wtime() - c0;
                pending = 0;
                if (recvbuf != NONE) {
                    found = recvbuf;
                    break;
                }
            }
            sendbuf = mine;
            MPI_Iallreduce(&sendbuf, &recvbuf, 1, MPI_UINT64_T, MPI_MIN,
                           MPI_COMM_WORLD, &req);
            pending = 1;
        }
        if (pending) {
            double c0 = MPI_Wtime();
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            comm += MPI_Wtime() - c0;
            if (found == NONE)
                found = recvbuf;
        }
    }
    double elapsed = MPI_Wtime() - t0;

    uint64_t all_typed = 0;
    double max_comm = 0.0;
    MPI_Reduce(&typed, &all_typed, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char host[256] = "unknown";
        gethostname(host, sizeof host - 1);
        uint64_t pos = (found == NONE) ? 0 : found;
        double wasted = pos ? 1.0 - (double)pos / (double)all_typed : 0.0;
        if (o.header)
            puts(FM_HEADER);
        printf("first_match,%s,%d,%d,%" PRIu64 ",%s,%d,%" PRIu64 ",%" PRIu64
               ",%s,%" PRIu64 ",%.6g,%" PRIu64 ",%.6f,%" PRIu64
               ",%.6f,%.6f,%.0f,%s,%s\n",
               kernel_name(o.kernel), size, threads, o.seed, targets.text[0],
               targets.len[0], o.chunk, o.check, o.async ? "async" : "sync",
               pos, monkey_expected_wait(targets.text[0], targets.len[0]),
               all_typed, wasted, rounds, elapsed, max_comm,
               (double)all_typed / elapsed, host, GIT_SHA);
    }
    MPI_Finalize();
    return 0;
}
