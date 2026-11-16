/*
 * cli.h - command line, timing and CSV output shared by every version
 * (serial, omp, mpi, hybrid), so all of them accept the same options and
 * print the same columns.
 */
#ifndef CLI_H
#define CLI_H

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "kernels.h"

#ifndef GIT_SHA
#define GIT_SHA "unknown"
#endif

#define CSV_HEADER                                                         \
    "version,kernel,ranks,threads,n,chunk,seed,target,len,count,expected," \
    "time_s,chars_per_s,host,git,busy_min_s,busy_max_s,comm_s"

typedef struct {
    uint64_t    n;      /* total characters typed                  */
    uint64_t    chunk;  /* characters per chunk (chunked kernel)   */
    uint64_t    seed;   /* run seed                                */
    const char *spec;   /* comma-separated targets                 */
    int         kernel; /* KERNEL_CHUNKED, KERNEL_FUSED, KERNEL_V0 */
    int         header; /* print CSV header first                  */
    /* first_match only */
    uint64_t    check;  /* chunks per thread between stop checks (-K) */
    int         async;  /* -m async: overlap the stop check (Iallreduce) */
} monkey_opts_t;

static inline double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * Parse a non-negative integer: plain digits (200000000), an integral float
 * (2e8) or a power of two (2^24).
 */
static inline int parse_u64(const char *s, uint64_t *out)
{
    char *end;
    errno = 0;
    if (s[0] == '2' && s[1] == '^') {
        long e = strtol(s + 2, &end, 10);
        if (errno || end == s + 2 || *end != '\0' || e < 0 || e > 63)
            return -1;
        *out = 1ULL << e;
        return 0;
    }
    if (*s != '\0' && strspn(s, "0123456789") == strlen(s)) {
        unsigned long long v = strtoull(s, &end, 10);
        if (errno)
            return -1;
        *out = (uint64_t)v;
        return 0;
    }
    double d = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !(d >= 0.0) || d > 9.0e18 ||
        d != floor(d))
        return -1;
    *out = (uint64_t)d;
    return 0;
}

static inline void monkey_usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [-n chars] [-t targets] [-s seed] [-c chunk] "
            "[-k chunked|fused|v0] [-H]\n"
            "  -n  characters to type (200000000, 2e8 or 2^28), default 1e8\n"
            "  -t  comma-separated targets (a-z, space or _),"
            " default a,the,love,romeo\n"
            "  -s  RNG seed, default 1\n"
            "  -c  characters per chunk, default 2^24\n"
            "  -k  kernel: chunked (default), fused (optimized) or v0 (one stream,"
            " serial only)\n"
            "  -H  print CSV header\n"
            "  first_match only:\n"
            "  -K  chunks per thread between stop checks, default 16\n"
            "  -m  sync (blocking check) or async (overlapped), default sync\n",
            prog);
}

/*
 * Parse options into o and targets into t.
 * Returns -1 to continue, otherwise the exit status the caller should return.
 */
static inline int monkey_parse_opts(int argc, char **argv, monkey_opts_t *o,
                                    monkey_targets_t *t)
{
    o->n = 100000000ULL;
    o->chunk = MONKEY_DEFAULT_CHUNK;
    o->seed = 1;
    o->spec = "a,the,love,romeo";
    o->kernel = KERNEL_CHUNKED;
    o->header = 0;
    o->check = 16;
    o->async = 0;

    int opt;
    while ((opt = getopt(argc, argv, "n:t:s:c:k:K:m:Hh")) != -1) {
        switch (opt) {
        case 'n':
            if (parse_u64(optarg, &o->n) || o->n == 0) {
                fprintf(stderr, "error: bad -n \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 't':
            o->spec = optarg;
            break;
        case 's':
            if (parse_u64(optarg, &o->seed)) {
                fprintf(stderr, "error: bad -s \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 'c':
            if (parse_u64(optarg, &o->chunk) || o->chunk < MONKEY_MAX_LEN) {
                fprintf(stderr, "error: bad -c \"%s\" (min %d)\n", optarg,
                        MONKEY_MAX_LEN);
                return 2;
            }
            break;
        case 'k':
            o->kernel = kernel_parse(optarg);
            if (o->kernel < 0) {
                fprintf(stderr, "error: bad -k \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 'K':
            if (parse_u64(optarg, &o->check) || o->check == 0) {
                fprintf(stderr, "error: bad -K \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 'm':
            if (strcmp(optarg, "sync") == 0)
                o->async = 0;
            else if (strcmp(optarg, "async") == 0)
                o->async = 1;
            else {
                fprintf(stderr, "error: bad -m \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 'H':
            o->header = 1;
            break;
        case 'h':
            monkey_usage(argv[0]);
            return 0;
        default:
            monkey_usage(argv[0]);
            return 2;
        }
    }
    if (optind < argc) {
        monkey_usage(argv[0]);
        return 2;
    }
    if (monkey_parse_targets(o->spec, t))
        return 2;
    return -1;
}

/*
 * Print one CSV row per target. busy_min/busy_max: shortest and longest time
 * any single worker (thread or rank x thread) spent computing; their ratio
 * measures load imbalance (serial: both equal elapsed). comm: time spent in
 * MPI communication (0 without MPI).
 */
static inline void monkey_print_csv(const char *version, int ranks,
                                    int threads, const monkey_opts_t *o,
                                    const monkey_targets_t *t,
                                    const uint64_t *counts, double elapsed,
                                    double busy_min, double busy_max,
                                    double comm)
{
    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    host[sizeof host - 1] = '\0';

    if (o->header)
        puts(CSV_HEADER);
    for (int k = 0; k < t->n; k++) {
        int L = t->len[k];
        int v0 = o->kernel == KERNEL_V0;
        double expected = v0 ? monkey_expected_count(o->n, L)
                             : monkey_expected_count_chunked(o->n, o->chunk, L);
        printf("%s,%s,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%d,%" PRIu64
               ",%.10g,%.6f,%.0f,%s,%s,%.6f,%.6f,%.6f\n",
               version, kernel_name(o->kernel), ranks, threads, o->n,
               v0 ? o->n : o->chunk, o->seed, t->text[k], L, counts[k],
               expected, elapsed, (double)o->n / elapsed, host, GIT_SHA,
               busy_min, busy_max, comm);
    }
    fflush(stdout);
}

#endif /* CLI_H */
