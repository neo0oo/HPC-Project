/*
 * rng.h - per-monkey pseudo-random number generator.
 *
 * xoshiro256** (Blackman & Vigna), seeded with splitmix64.
 * Independent streams come from the xoshiro256 jump function: stream k is the
 * base sequence advanced by k * 2^128 steps, so streams never overlap.
 */
#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* RNG_HD lets the same functions compile for CUDA (__host__ __device__) or
 * OpenACC (acc routine seq); plain C gets static inline. */
#ifndef RNG_HD
#define RNG_HD static inline
#endif

typedef struct {
    uint64_t s[4];
} rng_t;

RNG_HD uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

RNG_HD uint64_t rng_rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

/* Next 64 random bits. */
RNG_HD uint64_t rng_next(rng_t *r)
{
    uint64_t *s = r->s;
    const uint64_t result = rng_rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rng_rotl(s[3], 45);

    return result;
}

/* Advance the state by 2^128 steps. */
static inline void rng_jump(rng_t *r)
{
    static const uint64_t JUMP[4] = {
        0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
        0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & (1ULL << b)) {
                s0 ^= r->s[0];
                s1 ^= r->s[1];
                s2 ^= r->s[2];
                s3 ^= r->s[3];
            }
            rng_next(r);
        }
    }
    r->s[0] = s0;
    r->s[1] = s1;
    r->s[2] = s2;
    r->s[3] = s3;
}

/*
 * Seed generator r for (seed, stream_id). The same seed with different
 * stream ids gives non-overlapping sequences (one per thread/rank).
 * Cost is O(stream_id) jumps (256 steps each), negligible for thousands of
 * streams.
 */
static inline void rng_seed(rng_t *r, uint64_t seed, uint64_t stream_id)
{
    uint64_t sm = seed;
    for (int i = 0; i < 4; i++)
        r->s[i] = splitmix64(&sm);

    /* xoshiro must not start from the all-zero state. */
    if ((r->s[0] | r->s[1] | r->s[2] | r->s[3]) == 0)
        r->s[0] = 1;

    for (uint64_t k = 0; k < stream_id; k++)
        rng_jump(r);
}

/*
 * Seed generator r for chunk `chunk` of the run keyed by `seed`, in O(1).
 *
 * The state is splitmix64 outputs 4*chunk .. 4*chunk+3 of the sequence that
 * starts at `seed` (splitmix64 is a counter run through a bijective mixer, so
 * no two chunks ever receive the same state; this is the seeding method
 * recommended by the xoshiro authors). Each chunk then uses a short stretch
 * (<= 2^40 draws) of a 2^256-1 period, so overlap between chunks has
 * negligible probability.
 *
 * Chunk 0 gets exactly the state rng_seed(r, seed, 0) gives, so a run with a
 * single chunk reproduces the original single-stream (v0) kernel.
 */
RNG_HD void rng_seed_chunk(rng_t *r, uint64_t seed, uint64_t chunk)
{
    uint64_t sm = seed + 4 * chunk * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; i++)
        r->s[i] = splitmix64(&sm);

    if ((r->s[0] | r->s[1] | r->s[2] | r->s[3]) == 0)
        r->s[0] = 1;
}

#endif /* RNG_H */
