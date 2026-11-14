/*
 * serial.c - single-core baseline (T1).
 *
 * Usage: ./serial [-n chars] [-t targets] [-s seed] [-H]
 *   -n  number of characters to type (accepts 200000000 or 2e8), default 1e8
 *   -t  comma-separated targets (a-z, space or _), default a,the,love,romeo
 *   -s  RNG seed, default 1
 *   -H  print the CSV header line first
 *
 * Output: one CSV row per target:
 *   version,ranks,threads,n,seed,target,len,count,expected,time_s,chars_per_s
 * time_s is the kernel time (generation + counting for all targets together).
 */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "monkey.h"

#define CSV_HEADER \
    "version,ranks,threads,n,seed,target,len,count,expected,time_s,chars_per_s"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Parse a non-negative integer; plain digits or an integral float like 2e8. */
static int parse_u64(const char *s, uint64_t *out)
{
    char *end;
    errno = 0;
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

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [-n chars] [-t targets] [-s seed] [-H]\n"
            "  -n  characters to type (e.g. 200000000 or 2e8), default 1e8\n"
            "  -t  comma-separated targets, default a,the,love,romeo\n"
            "  -s  RNG seed, default 1\n"
            "  -H  print CSV header\n",
            prog);
}

int main(int argc, char **argv)
{
    uint64_t n = 100000000ULL, seed = 1;
    const char *spec = "a,the,love,romeo";
    int header = 0, opt;

    while ((opt = getopt(argc, argv, "n:t:s:Hh")) != -1) {
        switch (opt) {
        case 'n':
            if (parse_u64(optarg, &n) || n == 0) {
                fprintf(stderr, "error: bad -n \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 't':
            spec = optarg;
            break;
        case 's':
            if (parse_u64(optarg, &seed)) {
                fprintf(stderr, "error: bad -s \"%s\"\n", optarg);
                return 2;
            }
            break;
        case 'H':
            header = 1;
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 2;
        }
    }
    if (optind < argc) {
        usage(argv[0]);
        return 2;
    }

    monkey_targets_t targets;
    if (monkey_parse_targets(spec, &targets))
        return 2;

    uint64_t counts[MONKEY_MAX_TARGETS];
    double t0 = now_s();
    monkey_count(n, seed, 0, &targets, counts);
    double elapsed = now_s() - t0;

    if (header)
        puts(CSV_HEADER);
    for (int k = 0; k < targets.n; k++) {
        printf("serial,1,1,%" PRIu64 ",%" PRIu64 ",%s,%d,%" PRIu64
               ",%.10g,%.6f,%.0f\n",
               n, seed, targets.text[k], targets.len[k], counts[k],
               monkey_expected_count(n, targets.len[k]), elapsed,
               (double)n / elapsed);
    }
    return 0;
}
