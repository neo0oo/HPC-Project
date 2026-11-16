/*
 * kernels.h - one entry point for every kernel, used by all programs.
 *
 *   chunked  baseline: buffer + one rolling-hash pass per target (monkey.h)
 *   fused    optimized: shift register, all targets in one pass (fused.h)
 *   v0       original single stream, serial only (monkey.h)
 *
 * chunked and fused type identical characters, so their counts are
 * bit-identical for the same (n, chunk, seed).
 */
#ifndef KERNELS_H
#define KERNELS_H

#include "fused.h"
#include "monkey.h"

enum { KERNEL_CHUNKED = 0, KERNEL_V0 = 1, KERNEL_FUSED = 2 };

static inline const char *kernel_name(int k)
{
    return k == KERNEL_V0 ? "v0" : k == KERNEL_FUSED ? "fused" : "chunked";
}

static inline int kernel_parse(const char *s)
{
    if (strcmp(s, "chunked") == 0)
        return KERNEL_CHUNKED;
    if (strcmp(s, "v0") == 0)
        return KERNEL_V0;
    if (strcmp(s, "fused") == 0)
        return KERNEL_FUSED;
    return -1;
}

typedef struct {
    int              kernel;
    monkey_targets_t t;
    fused_targets_t  f;   /* prepared once, read-only, shared by threads */
} monkey_work_t;

static inline void work_init(monkey_work_t *w, int kernel,
                             const monkey_targets_t *t)
{
    w->kernel = kernel;
    w->t = *t;
    fused_prepare(t, &w->f);
}

/* Type and match chunk i of a run (chunked or fused), adding into counts. */
static inline void work_chunk(const monkey_work_t *w, uint64_t n,
                              uint64_t chunk, uint64_t seed, uint64_t i,
                              uint64_t *counts)
{
    rng_t r;
    rng_seed_chunk(&r, seed, i);
    uint64_t len = monkey_chunk_len(n, chunk, i);
    if (w->kernel == KERNEL_FUSED)
        fused_scan(&r, len, &w->f, counts);
    else
        monkey_scan(&r, len, &w->t, counts);
}

/*
 * First occurrence inside chunk i: number of characters typed in the chunk
 * when the first match completes (1-based), or 0. *typed receives how many
 * characters were actually generated (the kernel stops at the first match).
 * With the baseline kernel the whole chunk is counted first (that is its
 * cost); only a chunk that contains a match is rescanned for the position.
 */
static inline uint64_t work_chunk_first(const monkey_work_t *w, uint64_t len,
                                        uint64_t seed, uint64_t i,
                                        uint64_t *typed)
{
    rng_t r;
    rng_seed_chunk(&r, seed, i);
    if (w->kernel == KERNEL_FUSED) {
        uint64_t pos = fused_first(&r, len, &w->f);
        *typed = pos ? pos : len;
        return pos;
    }
    uint64_t counts[MONKEY_MAX_TARGETS] = {0};
    monkey_scan(&r, len, &w->t, counts);
    *typed = len;
    uint64_t total = 0;
    for (int k = 0; k < w->t.n; k++)
        total += counts[k];
    if (total == 0)
        return 0;
    rng_seed_chunk(&r, seed, i);
    return fused_first(&r, len, &w->f);
}

#endif /* KERNELS_H */
