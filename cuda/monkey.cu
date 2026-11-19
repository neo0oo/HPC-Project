/*
 * monkey.cu - CUDA version (additional technology).
 *
 *     ./monkey_cuda -n 1e12 -t romeo -c 2^16 -H
 *     CUDA_BLOCK=256 ./monkey_cuda ...      threads per block (default 256)
 *
 * Same command line and CSV as the CPU programs. Each GPU thread types whole
 * chunks (grid-stride loop over chunk ids) with the SAME generator and the
 * same fused shift-register matcher as the CPU (fused_chunk_simple, compiled
 * for the device through RNG_HD). So for the same (n, chunk, seed) the counts
 * are bit-identical to ./serial: checked with
 *     analysis/check_repro.py --bin bin/serial --compare bin/monkey_cuda
 *
 * GPUs need ~10^5-10^6 threads, so use small chunks (-c 2^16 or 2^18) to give
 * every thread several chunks. Per-thread counters live in registers (the
 * target count is a template parameter), are reduced with warp shuffles, and
 * one atomicAdd per warp writes the totals.
 *
 * time_s is measured with CUDA events around the kernel only; the
 * host<->device traffic is a few bytes (targets in, counters out).
 */
#define RNG_HD static __host__ __device__ __forceinline__
#include "../src/cli.h"

#include <cuda_runtime.h>

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e_ = (call);                                           \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(e_), __FILE__, __LINE__);           \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__constant__ uint64_t c_mask[MONKEY_MAX_TARGETS];
__constant__ uint64_t c_code[MONKEY_MAX_TARGETS];

template <int NT>
__global__ void count_kernel(uint64_t n, uint64_t chunk, uint64_t seed,
                             uint64_t nchunks, unsigned long long *out)
{
    uint64_t cnt[NT];
#pragma unroll
    for (int k = 0; k < NT; k++)
        cnt[k] = 0;

    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nchunks; i += stride) {
        uint64_t start = i * chunk;
        uint64_t len = (n - start < chunk) ? n - start : chunk;
        fused_chunk_simple(seed, i, len, NT, c_mask, c_code, cnt);
    }

    const unsigned lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < NT; k++) {
        unsigned long long v = cnt[k];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0 && v)
            atomicAdd(&out[k], v);
    }
}

template <int NT>
static float launch(uint64_t n, uint64_t chunk, uint64_t seed,
                    uint64_t nchunks, int block, unsigned long long *d_out,
                    long *threads_out)
{
    int dev, sms, per_sm = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, count_kernel<NT>, block, 0));
    uint64_t want = (nchunks + block - 1) / block;
    uint64_t grid = (uint64_t)sms * (per_sm > 0 ? per_sm : 1);
    if (want < grid)
        grid = want ? want : 1;
    *threads_out = (long)(grid * block);

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    count_kernel<NT><<<(unsigned)grid, block>>>(n, chunk, seed, nchunks, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

int main(int argc, char **argv)
{
    monkey_opts_t o;
    monkey_targets_t targets;
    int rc = monkey_parse_opts(argc, argv, &o, &targets);
    if (rc >= 0)
        return rc;
    if (o.kernel == KERNEL_V0) {
        fprintf(stderr, "error: v0 is serial only\n");
        return 2;
    }

    /* Pad the target list to a power of two with entries that never match
     * (a 5-bit field of 31 cannot occur). */
    fused_targets_t f;
    fused_prepare(&targets, &f);
    int nt = 1;
    while (nt < targets.n)
        nt *= 2;
    uint64_t mask[MONKEY_MAX_TARGETS], code[MONKEY_MAX_TARGETS];
    for (int k = 0; k < nt; k++) {
        mask[k] = k < targets.n ? f.mask[k] : 0x1F;
        code[k] = k < targets.n ? f.code[k] : 0x1F;
    }
    CHECK(cudaMemcpyToSymbol(c_mask, mask, sizeof mask));
    CHECK(cudaMemcpyToSymbol(c_code, code, sizeof code));

    unsigned long long *d_out;
    CHECK(cudaMalloc(&d_out, sizeof(unsigned long long) * MONKEY_MAX_TARGETS));
    CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long) * MONKEY_MAX_TARGETS));

    const char *b = getenv("CUDA_BLOCK");
    int block = b ? atoi(b) : 256;
    uint64_t nchunks = monkey_num_chunks(o.n, o.chunk);
    long threads = 0;
    float ms = 0;
    switch (nt) {
    case 1:  ms = launch<1>(o.n, o.chunk, o.seed, nchunks, block, d_out, &threads); break;
    case 2:  ms = launch<2>(o.n, o.chunk, o.seed, nchunks, block, d_out, &threads); break;
    case 4:  ms = launch<4>(o.n, o.chunk, o.seed, nchunks, block, d_out, &threads); break;
    case 8:  ms = launch<8>(o.n, o.chunk, o.seed, nchunks, block, d_out, &threads); break;
    default: ms = launch<16>(o.n, o.chunk, o.seed, nchunks, block, d_out, &threads); break;
    }

    unsigned long long out[MONKEY_MAX_TARGETS];
    CHECK(cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost));
    uint64_t counts[MONKEY_MAX_TARGETS];
    for (int k = 0; k < targets.n; k++)
        counts[k] = out[k];

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "# gpu=%s sm=%d.%d threads=%ld block=%d chunks=%llu\n",
            prop.name, prop.major, prop.minor, threads, block,
            (unsigned long long)nchunks);

    o.kernel = KERNEL_FUSED;
    double s = ms / 1000.0;
    monkey_print_csv("cuda", 1, (int)threads, &o, &targets, counts, s, s, s,
                     0.0);
    CHECK(cudaFree(d_out));
    return 0;
}
