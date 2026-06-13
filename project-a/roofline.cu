/*
 * Project A: Roofline plot for square FP32 and FP16 matmul on RTX 5060 laptop GPU
 *            (Lenovo Legion 5, Blackwell GB206).
 *
 * Goal: benchmark matmul at various sizes, compute arithmetic intensity,
 * and overlay measured TFLOPS on a roofline model.
 *
 * Roofline model recap:
 *   - x-axis: arithmetic intensity (FLOPS / byte)
 *   - y-axis: attainable FLOPS/s
 *   - Two ceilings: memory bandwidth bound, compute bound
 *   - Ridge point: where the two ceilings intersect
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -lcublas -o roofline roofline.cu
 *   (sm_120 = Blackwell / RTX 5060 laptop)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Hardware constants for RTX 5060 laptop GPU (Lenovo Legion 5, Blackwell GB206)
//
//   FP32 : 3840 CUDA cores × 2 ops/cycle × 2.497 GHz ≈ 19.2 TFLOPS
//   FP16 : Blackwell 5th-gen tensor cores, 8× FP32 dense  ≈ 153.6 TFLOPS
//   BW   : 128-bit GDDR7 @ ~17 Gbps                       ≈ 272 GB/s
//
//   NOTE: these are max-P (115 W) estimates; max-Q mode will be lower.
//   Verify with: nvidia-smi --query-gpu=clocks.max.sm,memory.total --format=csv
// ---------------------------------------------------------------------------
static constexpr double PEAK_FP32_TFLOPS = 19.2;   // theoretical peak FP32 TFLOPS
static constexpr double PEAK_FP16_TFLOPS = 153.6;  // theoretical peak FP16 TFLOPS (tensor cores)
static constexpr double MEM_BW_TB_S      = 0.272;  // memory bandwidth in TB/s

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));                 \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t _s = (call);                                               \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",                \
                    __FILE__, __LINE__, (int)_s);                                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// 2. Benchmark a single matmul
//    Returns: median elapsed time in seconds
// ---------------------------------------------------------------------------

/*
 * FP32: run C = A * B for N×N float matrices, return median time (s).
 *
 * Hints:
 *   - allocate A, B, C on device with cudaMalloc (each N*N floats)
 *   - optionally fill with curandGenerateUniform or cudaMemset
 *   - call cublasSgemm in a loop (warmup first, then timed iters)
 *       alpha=1, beta=0, OP_N for both A and B (column-major, so A×B works as-is)
 *   - time each iter with a pair of cudaEvent_t: Record before and after, then
 *       cudaEventSynchronize + cudaEventElapsedTime
 *   - collect elapsed times into a vector, sort, return the median / 1000.0 (ms→s)
 *   - free device buffers before returning
 */
double benchmark_matmul_fp32(cublasHandle_t handle, int N, int warmup = 5, int iters = 20) {
    //declaring memory allocations for the matrices
    float *A, *B, *C;
    cudaMalloc(&A, N * N * sizeof(float));
    cudaMalloc(&B, N * N * sizeof(float));
    cudaMalloc(&C, N * N * sizeof(float));

    //run warmup iterations for GPU 
    float alpha = 1.0f, beta = 0.0f;
    for(int i{}; i < warmup; ++i) {
         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    }

    //time each iteration with cuda events
    std::vector<float> times(iters);
    for (int i{}; i < iters; ++i) {
        //intializes start and stop var as time int?
        cudaEvent_t start, stop;
        //assigns it to the event
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[i], start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(times.begin(), times.end());
    double median = times[iters/2] / 1000.0;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return median;

    (void)handle; (void)N; (void)warmup; (void)iters;
    return 0.0;
}

/*
 * FP16: same structure as above but with __half buffers and cublasHgemm.
 *
 * Hints:
 *   - sizeof(__half) == 2; allocate N*N * sizeof(__half) bytes each
 *   - alpha/beta must also be __half: __float2half(1.0f) / __float2half(0.0f)
 *   - cublasHgemm signature mirrors cublasSgemm, just replace S→H and float*→__half*
 *   - alternatively use cublasGemmEx with CUDA_R_16F and CUBLAS_GEMM_DEFAULT_TENSOR_OP
 *     to explicitly request tensor-core paths
 */
double benchmark_matmul_fp16(cublasHandle_t handle, int N,
                              int warmup = 5, int iters = 20) {
    // TODO: implement
    (void)handle; (void)N; (void)warmup; (void)iters;
    return 0.0;
}

// ---------------------------------------------------------------------------
// 3. Compute arithmetic intensity for square matmul
// ---------------------------------------------------------------------------
struct MatmulStats {
    long long flops;
    long long bytes;
    double    arithmetic_intensity;  // FLOP / byte
};

/*
 * For C = A @ B with N×N matrices:
 *   flops = 2 * N^3          (N^3 multiplies + N^3 adds)
 *   bytes = 3 * N^2 * bytes_per_elem   (read A, read B, write C)
 *   arithmetic_intensity = (double)flops / bytes
 *
 * bytes_per_elem: 4 for FP32, 2 for FP16
 */
MatmulStats matmul_arithmetic_intensity(int N, int bytes_per_elem) {
    // TODO: implement
    (void)N; (void)bytes_per_elem;
    return {0, 0, 0.0};
}

// ---------------------------------------------------------------------------
// 4. Sweep matrix sizes and collect measurements
// ---------------------------------------------------------------------------
struct SweepResults {
    std::vector<int>    sizes;
    std::vector<double> tflops;
    std::vector<double> intensities;
};

/*
 * For each N in sizes, run benchmark_matmul_fp32 and compute:
 *   - stats   = matmul_arithmetic_intensity(N, 4)
 *   - elapsed = benchmark_matmul_fp32(handle, N)
 *   - measured_tflops = (double)stats.flops / elapsed / 1e12
 * Collect into SweepResults and return.
 */
SweepResults run_sweep_fp32(cublasHandle_t handle, const std::vector<int>& sizes) {
    // TODO: implement
    (void)handle;
    return {sizes, {}, {}};
}

/*
 * Same as run_sweep_fp32 but calls benchmark_matmul_fp16 and uses bytes_per_elem=2.
 */
SweepResults run_sweep_fp16(cublasHandle_t handle, const std::vector<int>& sizes) {
    // TODO: implement
    (void)handle;
    return {sizes, {}, {}};
}

// ---------------------------------------------------------------------------
// 5. Print roofline data (pipe to a plotting script or redirect to CSV)
// ---------------------------------------------------------------------------

/*
 * Emit two sections to stdout:
 *
 *   Section A — measured points (one per size):
 *     <label>,measured,<size>,<intensity>,<tflops>
 *
 *   Section B — roofline ceiling curve (sample ~100 x values log-spaced):
 *     <label>,ceiling,<x>,<attainable_tflops>
 *     where attainable_tflops = min(MEM_BW_TB_S * x, peak_tflops)
 *     and peak_tflops is PEAK_FP32_TFLOPS or PEAK_FP16_TFLOPS depending on label.
 *
 * Hints:
 *   - use logspace: x_i = x_min * pow(x_max/x_min, i/(n-1)) for i in [0,n)
 *   - ridge point is at peak_tflops / MEM_BW_TB_S (units: FLOP/byte)
 */
void print_results(const char* label, double peak_tflops, const SweepResults& r) {
    // TODO: implement
    (void)label; (void)peak_tflops; (void)r;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    printf("Benchmarking fp32...\n");
    SweepResults fp32_results = run_sweep_fp32(handle, sizes);

    printf("Benchmarking fp16...\n");
    SweepResults fp16_results = run_sweep_fp16(handle, sizes);

    // Print CSV to stdout; redirect to roofline.csv and plot separately
    printf("dtype,kind,size_or_x,intensity_or_x,tflops\n");
    print_results("fp32", PEAK_FP32_TFLOPS, fp32_results);
    print_results("fp16", PEAK_FP16_TFLOPS, fp16_results);

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
