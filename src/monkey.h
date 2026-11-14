/*
 * monkey.h - infinite monkey kernel and theory.
 *
 * Alphabet: 'a'..'z' -> 0..25, ' ' -> 26 (k = 27). In target strings '_' is
 * accepted as an alias for space, so "to_be" means "to be".
 *
 * A target of length L is encoded as a base-27 number, most significant digit
 * first. The kernel keeps one rolling window per target:
 *     w = w*27 - out*27^L + c   (mod 2^64)
 * which equals the base-27 value of the last L characters, because the true
 * value is always < 27^L <= 27^12 < 2^64. Overlapping occurrences are counted.
 */
#ifndef MONKEY_H
#define MONKEY_H

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rng.h"

#define MONKEY_K           27
#define MONKEY_SPACE       26
#define MONKEY_MAX_LEN     12
#define MONKEY_MAX_TARGETS 16
#define MONKEY_BLOCK       8192 /* characters generated per batch */

typedef struct {
    int      n;                                      /* number of targets  */
    int      len[MONKEY_MAX_TARGETS];                /* length L           */
    uint64_t code[MONKEY_MAX_TARGETS];               /* base-27 value      */
    char     text[MONKEY_MAX_TARGETS][MONKEY_MAX_LEN + 1]; /* canonical form */
} monkey_targets_t;

/* 27^e for 0 <= e <= 13 (exact in uint64). */
static inline uint64_t monkey_pow27(int e)
{
    uint64_t p = 1;
    while (e-- > 0)
        p *= MONKEY_K;
    return p;
}

/* Map a character to 0..26, or -1 if it is not in the alphabet. */
static inline int monkey_char_index(char c)
{
    unsigned char u = (unsigned char)tolower((unsigned char)c);
    if (u >= 'a' && u <= 'z')
        return u - 'a';
    if (u == ' ' || u == '_')
        return MONKEY_SPACE;
    return -1;
}

static inline char monkey_index_char(int i)
{
    return i == MONKEY_SPACE ? ' ' : (char)('a' + i);
}

/*
 * Parse comma-separated targets, e.g. "a,the,love,romeo".
 * Returns 0 on success, -1 on error (message printed to stderr).
 */
static inline int monkey_parse_targets(const char *spec, monkey_targets_t *t)
{
    memset(t, 0, sizeof *t);
    const char *p = spec;

    for (;;) {
        const char *end = strchr(p, ',');
        size_t L = end ? (size_t)(end - p) : strlen(p);

        if (L == 0) {
            fprintf(stderr, "error: empty target in \"%s\"\n", spec);
            return -1;
        }
        if (L > MONKEY_MAX_LEN) {
            fprintf(stderr, "error: target \"%.*s\" longer than %d chars\n",
                    (int)L, p, MONKEY_MAX_LEN);
            return -1;
        }
        if (t->n == MONKEY_MAX_TARGETS) {
            fprintf(stderr, "error: more than %d targets\n", MONKEY_MAX_TARGETS);
            return -1;
        }

        uint64_t code = 0;
        for (size_t j = 0; j < L; j++) {
            int c = monkey_char_index(p[j]);
            if (c < 0) {
                fprintf(stderr, "error: target \"%.*s\" has invalid char '%c' "
                        "(allowed: a-z, space, _)\n", (int)L, p, p[j]);
                return -1;
            }
            code = code * MONKEY_K + (uint64_t)c;
            t->text[t->n][j] = monkey_index_char(c);
        }
        t->text[t->n][L] = '\0';
        t->len[t->n] = (int)L;
        t->code[t->n] = code;
        t->n++;

        if (!end)
            break;
        p = end + 1;
    }
    return 0;
}

/* Uniform draw in 0..26 from 32 random bits (Lemire, with rejection). */
static inline uint8_t monkey_draw(uint32_t x, rng_t *r)
{
    uint64_t m = (uint64_t)x * MONKEY_K;
    uint32_t lo = (uint32_t)m;
    if (lo < MONKEY_K) {
        const uint32_t thresh = (uint32_t)(-MONKEY_K) % MONKEY_K;
        while (lo < thresh) {
            m = (uint64_t)(uint32_t)(rng_next(r) >> 32) * MONKEY_K;
            lo = (uint32_t)m;
        }
    }
    return (uint8_t)(m >> 32);
}

/* Fill buf with len random characters (values 0..26). */
static inline void monkey_fill(rng_t *r, uint8_t *buf, size_t len)
{
    size_t i = 0;
    for (; i + 2 <= len; i += 2) {
        uint64_t x = rng_next(r);
        buf[i]     = monkey_draw((uint32_t)x, r);
        buf[i + 1] = monkey_draw((uint32_t)(x >> 32), r);
    }
    if (i < len)
        buf[i] = monkey_draw((uint32_t)rng_next(r), r);
}

/*
 * Type n random characters from stream (seed, stream) and count overlapping
 * occurrences of every target. counts[k] receives the count for target k.
 * Only windows lying entirely inside the n characters are counted, so the
 * expectation is (n - L + 1) / 27^L.
 */
static inline void monkey_count(uint64_t n, uint64_t seed, uint64_t stream,
                                const monkey_targets_t *t, uint64_t *counts)
{
    rng_t r;
    rng_seed(&r, seed, stream);

    /* buf = [MONKEY_MAX_LEN chars carried from the previous block | block].
     * The initial carry is all zeros and every window starts at 0, which is
     * consistent (it is the value of L zero digits). Windows touching the
     * carry are skipped in the first block. */
    uint8_t buf[MONKEY_MAX_LEN + MONKEY_BLOCK] = {0};
    uint8_t *blk = buf + MONKEY_MAX_LEN;

    uint64_t w[MONKEY_MAX_TARGETS] = {0};
    uint64_t drop[MONKEY_MAX_TARGETS]; /* 27^L: weight of outgoing char */
    for (int k = 0; k < t->n; k++) {
        counts[k] = 0;
        drop[k] = monkey_pow27(t->len[k]);
    }

    uint64_t done = 0;
    while (done < n) {
        size_t m = (n - done < MONKEY_BLOCK) ? (size_t)(n - done) : MONKEY_BLOCK;
        monkey_fill(&r, blk, m);

        for (int k = 0; k < t->n; k++) {
            const int L = t->len[k];
            const uint64_t code = t->code[k], d = drop[k];
            const uint8_t *out = blk - L; /* out[i] leaves the window at i */
            uint64_t wk = w[k], c = 0;
            size_t i = 0;

            /* Warm-up: the first L-1 characters cannot end a full window. */
            if (done < (uint64_t)(L - 1)) {
                size_t warm = (size_t)(L - 1 - done);
                if (warm > m)
                    warm = m;
                for (; i < warm; i++)
                    wk = wk * MONKEY_K - out[i] * d + blk[i];
            }
            for (; i < m; i++) {
                wk = wk * MONKEY_K - out[i] * d + blk[i];
                c += (wk == code);
            }
            w[k] = wk;
            counts[k] += c;
        }

        /* Carry the last MONKEY_MAX_LEN characters into the next block. */
        memmove(buf, blk + m - MONKEY_MAX_LEN, MONKEY_MAX_LEN);
        done += m;
    }
}

/* Expected number of (overlapping) occurrences in n characters. */
static inline double monkey_expected_count(uint64_t n, int L)
{
    if (n < (uint64_t)L)
        return 0.0;
    return (double)(n - (uint64_t)L + 1) / (double)monkey_pow27(L);
}

/*
 * Expected number of characters typed until the target first appears
 * (Conway's overlap formula): sum of 27^j over every j in 1..L for which the
 * length-j prefix equals the length-j suffix. "the" -> 27^3, "aa" -> 27 + 27^2.
 */
static inline double monkey_expected_wait(const char *text, int L)
{
    double e = 0.0;
    for (int j = 1; j <= L; j++)
        if (memcmp(text, text + L - j, (size_t)j) == 0)
            e += (double)monkey_pow27(j);
    return e;
}

#endif /* MONKEY_H */
