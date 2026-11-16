/*
 * fused.h - optimized matching kernel ("fused").
 *
 * The baseline (monkey_scan) writes random characters into a buffer and then
 * makes ONE PASS PER TARGET over it, each pass carrying the rolling value
 *     w = w*27 - out*27^L + c
 * so every character costs ~0.96 ns per target, a multiply-add dependency
 * chain (measured: t = 0.75 + 0.96*k ns/char, docs/FINDINGS.md).
 *
 * The fused kernel instead:
 *  1. keeps the last 12 characters in a 64-bit SHIFT REGISTER, 5 bits per
 *     character (27 symbols < 32):  w = (w << 5) | c.
 *     Two characters are folded per step, w' = (w << 10) | (c0 << 5) | c1,
 *     so the loop-carried chain is one shift + or per TWO characters.
 *  2. tests ALL targets in the same pass. Target k of length L matches when
 *     (w & mask_L) == code_k. Targets of the same length share one key
 *     (w & mask_L); with more than FUSED_DIRECT targets of a length the key
 *     is first tested against a 4096-bit bitmap (one hash + one bit test per
 *     length), and only bitmap hits are compared exactly. So the cost grows
 *     with the number of distinct LENGTHS, not the number of targets.
 *  3. needs no buffer and no warm-up: w starts as all ones, and 5-bit fields
 *     27..31 never occur in real text, so a window that still contains
 *     initial bits can never equal a target code.
 *
 * It draws exactly the same characters as the baseline (same generator,
 * same Lemire draws, same order), so for the same (n, chunk, seed) its counts
 * are bit-identical to the "chunked" kernel. That is the correctness test.
 *
 * fused_chunk_simple() is the portable form (no bitmap) used by the GPU
 * versions (CUDA, OpenACC); it gives the same counts.
 */
#ifndef FUSED_H
#define FUSED_H

#include "monkey.h"

#define FUSED_DIRECT 2      /* <= this many targets of one length: compare */
#define FUSED_BITS   12     /* bitmap of 2^12 bits per length group        */
#define FUSED_GROUPS MONKEY_MAX_LEN

typedef struct {
    int      n;                          /* targets                        */
    uint64_t code[MONKEY_MAX_TARGETS];   /* 5-bit packed target            */
    uint64_t mask[MONKEY_MAX_TARGETS];   /* low 5*L bits                   */

    int      ngroups;                    /* distinct target lengths        */
    uint64_t gmask[FUSED_GROUPS];
    int      gsize[FUSED_GROUPS];
    int      gmember[FUSED_GROUPS][MONKEY_MAX_TARGETS];
    uint64_t bitmap[FUSED_GROUPS][(1 << FUSED_BITS) / 64];
} fused_targets_t;

static inline uint64_t fused_mask(int L)
{
    return (L >= 12) ? ((1ULL << 60) - 1) : ((1ULL << (5 * L)) - 1);
}

static inline uint32_t fused_hash(uint64_t key)
{
    return (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - FUSED_BITS));
}

static inline void fused_prepare(const monkey_targets_t *t, fused_targets_t *f)
{
    memset(f, 0, sizeof *f);
    f->n = t->n;
    for (int k = 0; k < t->n; k++) {
        uint64_t code = 0;
        for (int j = 0; j < t->len[k]; j++)
            code = (code << 5) | (uint64_t)monkey_char_index(t->text[k][j]);
        f->code[k] = code;
        f->mask[k] = fused_mask(t->len[k]);

        int g = 0;
        while (g < f->ngroups && f->gmask[g] != f->mask[k])
            g++;
        if (g == f->ngroups) {
            f->gmask[g] = f->mask[k];
            f->ngroups++;
        }
        f->gmember[g][f->gsize[g]++] = k;
        uint32_t h = fused_hash(code);
        f->bitmap[g][h >> 6] |= 1ULL << (h & 63);
    }
}

/* Count matches of window value w for every target (adds into counts). */
static inline void fused_match(const fused_targets_t *f, uint64_t w,
                               uint64_t *counts)
{
    for (int g = 0; g < f->ngroups; g++) {
        const uint64_t key = w & f->gmask[g];
        const int size = f->gsize[g];
        if (size <= FUSED_DIRECT) {
            for (int m = 0; m < size; m++) {
                int k = f->gmember[g][m];
                counts[k] += (key == f->code[k]);
            }
        } else {
            uint32_t h = fused_hash(key);
            if ((f->bitmap[g][h >> 6] >> (h & 63)) & 1) { /* rare */
                for (int m = 0; m < size; m++) {
                    int k = f->gmember[g][m];
                    counts[k] += (key == f->code[k]);
                }
            }
        }
    }
}

/* 1 if window value w matches any target. */
static inline int fused_any(const fused_targets_t *f, uint64_t w)
{
    for (int k = 0; k < f->n; k++)
        if ((w & f->mask[k]) == f->code[k])
            return 1;
    return 0;
}

/*
 * Type up to n characters from r; return the number of characters typed when
 * the first occurrence of any target is complete (1-based), or 0 if none.
 */
static inline uint64_t fused_first(rng_t *r, uint64_t n,
                                   const fused_targets_t *f)
{
    uint64_t w = ~0ULL, i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64_t x = rng_next(r);
        uint64_t c0 = monkey_draw((uint32_t)x, r);
        uint64_t c1 = monkey_draw((uint32_t)(x >> 32), r);
        uint64_t w0 = (w << 5) | c0;
        uint64_t w1 = (w << 10) | (c0 << 5) | c1;
        if (fused_any(f, w0))
            return i + 1;
        if (fused_any(f, w1))
            return i + 2;
        w = w1;
    }
    if (i < n) {
        w = (w << 5) | monkey_draw((uint32_t)rng_next(r), r);
        if (fused_any(f, w))
            return i + 1;
    }
    return 0;
}

/* Type n characters from r and add matches into counts (CPU, with bitmap). */
static inline void fused_scan(rng_t *r, uint64_t n, const fused_targets_t *f,
                              uint64_t *counts)
{
    uint64_t w = ~0ULL, i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64_t x = rng_next(r);
        uint64_t c0 = monkey_draw((uint32_t)x, r);
        uint64_t c1 = monkey_draw((uint32_t)(x >> 32), r);
        uint64_t w0 = (w << 5) | c0;
        uint64_t w1 = (w << 10) | (c0 << 5) | c1;
        fused_match(f, w0, counts);
        fused_match(f, w1, counts);
        w = w1;
    }
    if (i < n) {
        w = (w << 5) | monkey_draw((uint32_t)rng_next(r), r);
        fused_match(f, w, counts);
    }
}

/*
 * Portable form for accelerators: chunk `id` of a run, `len` characters,
 * targets as (mask, code) arrays, plain compare loop. Same counts as above.
 */
RNG_HD void fused_chunk_simple(uint64_t seed, uint64_t id, uint64_t len,
                               int nt, const uint64_t *mask,
                               const uint64_t *code, uint64_t *counts)
{
    rng_t r;
    rng_seed_chunk(&r, seed, id);
    uint64_t w = ~0ULL, i = 0;
    for (; i + 2 <= len; i += 2) {
        uint64_t x = rng_next(&r);
        uint64_t c0 = monkey_draw((uint32_t)x, &r);
        uint64_t c1 = monkey_draw((uint32_t)(x >> 32), &r);
        uint64_t w0 = (w << 5) | c0;
        uint64_t w1 = (w << 10) | (c0 << 5) | c1;
        for (int k = 0; k < nt; k++)
            counts[k] += ((w0 & mask[k]) == code[k]) + ((w1 & mask[k]) == code[k]);
        w = w1;
    }
    if (i < len) {
        w = (w << 5) | monkey_draw((uint32_t)rng_next(&r), &r);
        for (int k = 0; k < nt; k++)
            counts[k] += ((w & mask[k]) == code[k]);
    }
}

#endif /* FUSED_H */
