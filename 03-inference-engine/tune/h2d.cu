/*
 * Step 0: the two bandwidths the roofline is built on.
 *
 * PCIE_BW_GB_S is the second slope of the whole project and it was a comment in
 * roofline2.py until this file existed. It is not a spec-sheet number: this
 * laptop negotiates a Gen4 x8 link, WSL2 sits between the process and the
 * driver, and pinned vs pageable is worth a lot less here than the usual advice
 * assumes. All of that only shows up if you measure it.
 *
 * Also reports what the device says about itself: VRAM capacity sets where the
 * cliff falls, and bus width x memory clock is PEAK_BW_GB_S, the ceiling the
 * resident tier is quoted as a fraction of.
 *
 * Emits JSON on stdout for measure_machine.py.
 *
 *   nvcc -O3 -std=c++17 -arch=sm_120 -o h2d tune/h2d.cu && ./h2d
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

static constexpr int WARMUP_ITERS = 3;
static constexpr int TIMED_ITERS = 20;

// Big enough that the fixed per-transfer overhead stops mattering, small enough
// to leave room on an 8 GB card. The knee is the interesting part of the curve.
static const std::vector<size_t> SIZES_MIB = {1, 4, 16, 64, 256, 512};

struct Result {
    size_t mib;
    double h2d_pinned;
    double h2d_pageable;
    double d2h_pinned;
};

static double time_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; ++i)
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i)
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms{};
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (double)bytes * TIMED_ITERS / (ms * 1e-3) / 1e9;
}

static Result measure(size_t mib) {
    size_t bytes = mib << 20;

    void* d_buf{};
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));

    void* h_pinned{};
    CUDA_CHECK(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault));
    void* h_pageable = malloc(bytes);
    if (!h_pageable) { fprintf(stderr, "malloc %zu MiB failed\n", mib); exit(1); }

    Result r{};
    r.mib = mib;
    r.h2d_pinned = time_copy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice);
    r.h2d_pageable = time_copy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice);
    r.d2h_pinned = time_copy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost);

    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_buf));
    return r;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // GDDR is double data rate, and nvidia reports the clock already doubled on
    // some parts and not others, so this can disagree with the spec sheet. The
    // roofline uses the spec number for that reason; this is a cross-check.
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    std::vector<Result> results;
    for (size_t mib : SIZES_MIB)
        results.push_back(measure(mib));

    double best_h2d_pinned = 0, best_h2d_pageable = 0, best_d2h_pinned = 0;
    for (const Result& r : results) {
        best_h2d_pinned = std::max(best_h2d_pinned, r.h2d_pinned);
        best_h2d_pageable = std::max(best_h2d_pageable, r.h2d_pageable);
        best_d2h_pinned = std::max(best_d2h_pinned, r.d2h_pinned);
    }

    printf("{\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"sm_count\": %d,\n", prop.multiProcessorCount);
    printf("  \"vram_capacity_gb\": %.2f,\n", prop.totalGlobalMem / 1e9);
    printf("  \"memory_bus_bits\": %d,\n", prop.memoryBusWidth);
    printf("  \"vram_peak_bw_gb_s_from_clocks\": %.1f,\n", peak_bw);
    printf("  \"h2d_pinned_gb_s\": %.2f,\n", best_h2d_pinned);
    printf("  \"h2d_pageable_gb_s\": %.2f,\n", best_h2d_pageable);
    printf("  \"d2h_pinned_gb_s\": %.2f,\n", best_d2h_pinned);
    printf("  \"transfer_sweep\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        printf("    {\"mib\": %zu, \"h2d_pinned\": %.2f, \"h2d_pageable\": %.2f, \"d2h_pinned\": %.2f}%s\n",
               r.mib, r.h2d_pinned, r.h2d_pageable, r.d2h_pinned,
               i + 1 < results.size() ? "," : "");
    }
    printf("  ]\n");
    printf("}\n");
    return 0;
}
