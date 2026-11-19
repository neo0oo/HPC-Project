/*
 * serial.c - single-core baseline (T1).
 *
 * Usage: ./serial [-n chars] [-t targets] [-s seed] [-c chunk] [-k kernel] [-H]
 * (see cli.h). Output: one CSV row per target, columns in CSV_HEADER.
 * time_s is the kernel time (generation + counting for all targets together).
 *
 * -k chunked (default): n is typed as independent chunks, so the counts are
 *    identical to every parallel version for the same (n, chunk, seed).
 * -k fused: optimized single-pass kernel, same characters, same counts.
 * -k v0: the original single continuous stream, kept as the reference for
 *    measuring the cost of the chunked decomposition.
 */
#include "cli.h"

int main(int argc, char **argv)
{
    monkey_opts_t o;
    monkey_targets_t targets;
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0)
        return rc;

    monkey_work_t work;
    work_init(&work, o.kernel, &targets);

    uint64_t counts[MONKEY_MAX_TARGETS] = {0};
    double t0 = now_s();
    if (o.kernel == KERNEL_V0) {
        monkey_count(o.n, o.seed, 0, &targets, counts);
    } else {
        uint64_t nchunks = monkey_num_chunks(o.n, o.chunk);
        for (uint64_t i = 0; i < nchunks; i++)
            work_chunk(&work, o.n, o.chunk, o.seed, i, counts);
    }
    double elapsed = now_s() - t0;

    monkey_print_csv("serial", 1, 1, &o, &targets, counts, elapsed, elapsed,
                     elapsed, 0.0);
    return 0;
}
