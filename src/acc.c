/*
 * acc.c - OpenACC port of the counting kernel (technology exploration).
 *
 * Built with the NVIDIA HPC compiler (module nvhpc/25.7), twice:
 *   bin/acc_gpu        nvc -acc=gpu        (offload to the GPU)
 *   bin/acc_multicore  nvc -acc=multicore  (same source on all CPU cores)
 *
 * One directive parallelizes the loop over chunks; every chunk runs the
 * portable fused kernel (fused_chunk_simple), compiled for the device through
 * RNG_HD = "acc routine seq". Up to 4 targets (one scalar reduction each).
 * Counts must be identical to ./serial for the same (n, chunk, seed); compare
 * the speed with the hand-written CUDA kernel.
 */
#define RNG_HD _Pragma("acc routine seq") static inline
#include "cli.h"

#ifndef ACC_VERSION
#define ACC_VERSION "acc"
#endif

int main(int argc, char **argv)
{
    monkey_opts_t o;
    monkey_targets_t targets;
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0)
        return rc;
    if (targets.n > 4 || o.kernel == KERNEL_V0) {
        fprintf(stderr, "error: acc supports 1-4 targets, chunked/fused\n");
        return 2;
    }
    fused_targets_t f;
    fused_prepare(&targets, &f);
    uint64_t mask[4], code[4];
    for (int k = 0; k < 4; k++) {
        mask[k] = k < targets.n ? f.mask[k] : 0x1F; /* never matches */
        code[k] = k < targets.n ? f.code[k] : 0x1F;
    }

    const uint64_t n = o.n, chunk = o.chunk, seed = o.seed;
    const uint64_t nchunks = monkey_num_chunks(n, chunk);
    uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    /* Warm-up region: device/context initialization is not part of the
     * measured time (the CUDA version excludes it too). */
    int warm = 0;
#pragma acc parallel loop reduction(+ : warm)
    for (int i = 0; i < 1024; i++)
        warm += i & 1;
    if (warm < 0)
        return 1;

    double t0 = now_s();
#pragma acc parallel loop gang vector reduction(+ : c0, c1, c2, c3) \
    copyin(mask[0:4], code[0:4])
    for (uint64_t i = 0; i < nchunks; i++) {
        uint64_t cnt[4] = {0, 0, 0, 0};
        uint64_t start = i * chunk;
        uint64_t len = (n - start < chunk) ? n - start : chunk;
        fused_chunk_simple(seed, i, len, 4, mask, code, cnt);
        c0 += cnt[0];
        c1 += cnt[1];
        c2 += cnt[2];
        c3 += cnt[3];
    }
    double elapsed = now_s() - t0;

    uint64_t counts[MONKEY_MAX_TARGETS] = {c0, c1, c2, c3};
    o.kernel = KERNEL_FUSED;
    monkey_print_csv(ACC_VERSION, 1, 0, &o, &targets, counts, elapsed,
                     elapsed, elapsed, 0.0);
    return 0;
}
