/*
 * prof.c - single-core profile of each kernel.
 *
 *     ./prof -n 2e9 -t romeo -k chunked -H
 *     ./prof -n 2e9 -t romeo -k gen        (generate characters only)
 *
 * Built twice by the Makefile:
 *   bin/prof        timing only (works everywhere)
 *   bin/prof_papi   + hardware counters via PAPI (-DUSE_PAPI -lpapi)
 *
 * "gen" types the same characters as the kernels but matches nothing, so
 *     matching cost = kernel time - gen time
 * is measured directly (application instrumentation). With PAPI, every
 * kernel also reports instructions, cycles, IPC, branch mispredictions and
 * L1 data-cache misses per character: the evidence for what bounds it.
 * Counters that this CPU/PAPI build does not support are left empty.
 */
#include "cli.h"

#ifdef USE_PAPI
#include <papi.h>
static const char *EVNAMES[] = {"PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_BR_MSP",
                                "PAPI_L1_DCM"};
#define NEV 4
#endif

#define PROF_HEADER                                                         \
    "kernel,ntargets,n,time_s,ns_per_char,ins_per_char,cyc_per_char,ipc,"   \
    "br_msp_per_kchar,l1_dcm_per_kchar,host,git"

static volatile uint64_t sink; /* keeps "gen" from being optimized away */

static void gen_only(uint64_t n, uint64_t chunk, uint64_t seed)
{
    uint8_t buf[MONKEY_BLOCK];
    uint64_t acc = 0;
    for (uint64_t i = 0, nc = monkey_num_chunks(n, chunk); i < nc; i++) {
        rng_t r;
        rng_seed_chunk(&r, seed, i);
        uint64_t len = monkey_chunk_len(n, chunk, i);
        for (uint64_t done = 0; done < len; done += MONKEY_BLOCK) {
            size_t m = len - done < MONKEY_BLOCK ? len - done : MONKEY_BLOCK;
            monkey_fill(&r, buf, m);
            acc += buf[m - 1];
        }
    }
    sink = acc;
}

int main(int argc, char **argv)
{
    /* -k gen is handled here; everything else goes to the shared parser. */
    int gen = 0;
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], "-k") == 0 && strcmp(argv[i + 1], "gen") == 0) {
            gen = 1;
            argv[i + 1] = "chunked";
        }

    monkey_opts_t o;
    monkey_targets_t targets;
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0)
        return rc;
    monkey_work_t work;
    work_init(&work, o.kernel, &targets);

    double ev[4] = {-1, -1, -1, -1};
#ifdef USE_PAPI
    int es = PAPI_NULL, have[NEV] = {0}, nadded = 0;
    long long vals[NEV] = {0};
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT ||
        PAPI_create_eventset(&es) != PAPI_OK) {
        fprintf(stderr, "papi: init failed\n");
        es = PAPI_NULL;
    } else {
        for (int e = 0; e < NEV; e++) {
            int code;
            if (PAPI_event_name_to_code((char *)EVNAMES[e], &code) == PAPI_OK &&
                PAPI_add_event(es, code) == PAPI_OK) {
                have[e] = 1;
                nadded++;
            } else {
                fprintf(stderr, "papi: %s not available\n", EVNAMES[e]);
            }
        }
        if (nadded == 0 || PAPI_start(es) != PAPI_OK) {
            fprintf(stderr, "papi: cannot start counters "
                            "(check /proc/sys/kernel/perf_event_paranoid)\n");
            es = PAPI_NULL;
        }
    }
#endif

    uint64_t counts[MONKEY_MAX_TARGETS] = {0};
    double t0 = now_s();
    if (gen) {
        gen_only(o.n, o.chunk, o.seed);
    } else {
        for (uint64_t i = 0, nc = monkey_num_chunks(o.n, o.chunk); i < nc; i++)
            work_chunk(&work, o.n, o.chunk, o.seed, i, counts);
    }
    double elapsed = now_s() - t0;

#ifdef USE_PAPI
    if (es != PAPI_NULL) {
        long long raw[NEV] = {0};
        PAPI_stop(es, raw);
        for (int e = 0, j = 0; e < NEV; e++)
            if (have[e])
                vals[e] = raw[j++];
        for (int e = 0; e < NEV; e++)
            ev[e] = have[e] ? (double)vals[e] : -1;
    }
#endif

    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    double nc = (double)o.n;
    char ins[32] = "", cyc[32] = "", ipc[32] = "", br[32] = "", l1[32] = "";
    if (ev[0] >= 0) snprintf(ins, sizeof ins, "%.3f", ev[0] / nc);
    if (ev[1] >= 0) snprintf(cyc, sizeof cyc, "%.3f", ev[1] / nc);
    if (ev[0] >= 0 && ev[1] > 0) snprintf(ipc, sizeof ipc, "%.3f", ev[0] / ev[1]);
    if (ev[2] >= 0) snprintf(br, sizeof br, "%.3f", ev[2] / nc * 1000);
    if (ev[3] >= 0) snprintf(l1, sizeof l1, "%.3f", ev[3] / nc * 1000);

    if (o.header)
        puts(PROF_HEADER);
    printf("%s,%d,%" PRIu64 ",%.6f,%.4f,%s,%s,%s,%s,%s,%s,%s\n",
           gen ? "gen" : kernel_name(o.kernel), gen ? 0 : targets.n, o.n,
           elapsed, elapsed / nc * 1e9, ins, cyc, ipc, br, l1, host, GIT_SHA);
    return 0;
}
