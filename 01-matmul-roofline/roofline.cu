/*
 * Project A: Roofline plot for square FP32 and FP16 matmul on RTX 5060 laptop GPU
 *            (Lenovo Legion 5, Blackwell GB206).
 *
 * Build: nvcc -O3 -arch=sm_120 -lcublas -o roofline roofline.cu
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE_DIM 32

// Max-P (115 W) estimates; max-Q mode will be lower.
//   FP32 : 3840 CUDA cores × 2 ops/cycle × 2.497 GHz ≈ 19.2 TFLOPS
//   FP16 : Blackwell 5th-gen tensor cores, 8× FP32 dense  ≈ 153.6 TFLOPS
//   BW   : 128-bit GDDR7 @ ~17 Gbps                       ≈ 272 GB/s
static constexpr double PEAK_FP32_TFLOPS = 19.2;
static constexpr double PEAK_FP16_TFLOPS = 153.6;
static constexpr double MEM_BW_TB_S      = 0.272;

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

// returns median elapsed time in seconds
double benchmark_matmul_fp32(cublasHandle_t handle, int N, int warmup = 5, int iters = 20) {
    float *A, *B, *C;
    cudaMalloc(&A, N * N * sizeof(float));
    cudaMalloc(&B, N * N * sizeof(float));
    cudaMalloc(&C, N * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    for(int i{}; i < warmup; ++i) {
         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(iters);
    for (int i{}; i < iters; ++i) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[i], start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(times.begin(), times.end());
    double median = times[iters/2] / 1000.0;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return median;
}

double benchmark_matmul_fp16(cublasHandle_t handle, int N,
                              int warmup = 5, int iters = 20) {
    __half *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(__half)));

    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    for (int i = 0; i < warmup; ++i)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                  &alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N,
                                  &beta,  C, CUDA_R_16F, N,
                                  CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                  &alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N,
                                  &beta,  C, CUDA_R_16F, N,
                                  CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    double median = times[iters / 2] / 1000.0;
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return median;
}

struct MatmulStats {
    long long flops;
    long long bytes;
    double    arithmetic_intensity;  // FLOP / byte
};

// flops = 2*N^3 (N^3 multiplies + N^3 adds); bytes = 3*N^2*elem (read A, B, write C)
MatmulStats matmul_arithmetic_intensity(int N, int bytes_per_elem) {
    long long flops = 2LL * N * N * N;
    long long bytes = 3LL * N * N * bytes_per_elem;
    return {flops, bytes, (double)flops / bytes};
}

struct SweepResults {
    std::vector<int>    sizes;
    std::vector<double> tflops;
    std::vector<double> intensities;
};

SweepResults run_sweep_fp32(cublasHandle_t handle, const std::vector<int>& sizes) {
    SweepResults r;
    r.sizes = sizes;
    for (int N : sizes) {
        fprintf(stderr, "  fp32 N=%d...\n", N);
        MatmulStats s = matmul_arithmetic_intensity(N, 4);
        double elapsed = benchmark_matmul_fp32(handle, N);
        r.intensities.push_back(s.arithmetic_intensity);
        r.tflops.push_back((double)s.flops / elapsed / 1e12);
    }
    return r;
}

SweepResults run_sweep_fp16(cublasHandle_t handle, const std::vector<int>& sizes) {
    SweepResults r;
    r.sizes = sizes;
    for (int N : sizes) {
        fprintf(stderr, "  fp16 N=%d...\n", N);
        MatmulStats s = matmul_arithmetic_intensity(N, 2);
        double elapsed = benchmark_matmul_fp16(handle, N);
        r.intensities.push_back(s.arithmetic_intensity);
        r.tflops.push_back((double)s.flops / elapsed / 1e12);
    }
    return r;
}

// Emits measured points (<label>,measured,<size>,<intensity>,<tflops>) followed by
// the ceiling curve (<label>,ceiling,0,<x>,min(MEM_BW_TB_S * x, peak_tflops))
void print_results(const char* label, double peak_tflops, const SweepResults& r) {
    if (r.tflops.empty()) return;
    // Measured points
    for (size_t i = 0; i < r.sizes.size(); i++)
        printf("%s,measured,%d,%.6f,%.6f\n",
               label, r.sizes[i], r.intensities[i], r.tflops[i]);

    // Roofline ceiling curve (100 log-spaced x values)
    const int n = 100;
    double x_min = 0.1, x_max = 1000.0;
    for (int i = 0; i < n; i++) {
        double x = x_min * pow(x_max / x_min, (double)i / (n - 1));
        double attainable = std::min(MEM_BW_TB_S * x, peak_tflops);
        printf("%s,ceiling,0,%.6f,%.6f\n", label, x, attainable);
    }
}

__global__ void matmul_naive_fp32(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float output{};
        for (int j{}; j < N; ++j) {
            output += A[row * N + j] * B[j * N + col];
        }

        C[row * N + col] = output;
    }
}

__global__ void matmul_naive_fp16(__half *A, __half *B, __half *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float output = 0.0f;
        for (int j{}; j < N; ++j) {
            output += __half2float(A[row * N + j]) * __half2float(B[j * N + col]);
        }

        C[row * N + col] = __float2half(output);
    }
}


//Current optimized uses shared tiling only, still 1 kernel = 1 output, but faster memory access
//Future: optimize further by implementing register blocking, 1 kernel = multiple outputs

__global__ void matmul_optimized_fp32(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    float value{};

    for(int i{}; i < (N + TILE_DIM - 1) / TILE_DIM; ++i) {
        if (row < N && (i * TILE_DIM + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_DIM + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (i * TILE_DIM + threadIdx.y) < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_DIM + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        if (row < N && col < N) {
            for(int k{}; k < TILE_DIM; ++k) {
                value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

__global__ void matmul_optimized_fp16(__half *A, __half *B, __half *C, int N) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ __half tile_A[TILE_DIM][TILE_DIM];
    __shared__ __half tile_B[TILE_DIM][TILE_DIM];

    float value = 0.0f;

    for(int i{}; i < (N + TILE_DIM - 1) / TILE_DIM; ++i) {
        if(row < N && (i * TILE_DIM + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_DIM + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = (__half)0.0f;
        }

        if (col < N && (i * TILE_DIM + threadIdx.y) < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_DIM + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = (__half)0.0f;
        }

        __syncthreads();

        if (row < N && col < N) {
            for(int k{}; k < TILE_DIM; ++k) {
                value += __half2float(tile_A[threadIdx.y][k]) * __half2float(tile_B[k][threadIdx.x]);
            }
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = __float2half(value);
    }
}

double benchmark_matmul_fp32_naive(int N, int warmup = 5,int iters = 20) {
    float *A, *B, *C;

    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemset(A, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(B, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(float)));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i{}; i < warmup; ++i) {
        matmul_naive_fp32<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());      // catch bad launch config
    CUDA_CHECK(cudaDeviceSynchronize()); // catch errors during execution

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for(int i{}; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_naive_fp32<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    double median = times[iters / 2] / 1000.0;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return median;
}

double benchmark_matmul_fp16_naive (int N, int warmup = 5, int iters = 20) {
    __half *A, *B, *C;

    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(__half)));

    CUDA_CHECK(cudaMemset(A, 0, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(B, 0, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(__half)));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i{}; i < warmup; ++i) {
        matmul_naive_fp16<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());      // catch bad launch config
    CUDA_CHECK(cudaDeviceSynchronize()); // catch errors during execution

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for(int i{}; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_naive_fp16<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    std::sort(times.begin(), times.end());
    double median = times[iters / 2] / 1000.0;

    return median;

}

double benchmark_matmul_fp32_optimized(int N, int warmup = 5,int iters = 20) {
    float *A, *B, *C;

    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemset(A, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(B, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(float)));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i{}; i < warmup; ++i) {
        matmul_optimized_fp32<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());      // catch bad launch config
    CUDA_CHECK(cudaDeviceSynchronize()); // catch errors during execution

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for(int i{}; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_optimized_fp32<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    double median = times[iters / 2] / 1000.0;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return median;
}

double benchmark_matmul_fp16_optimized (int N, int warmup = 5, int iters = 20) {
    __half *A, *B, *C;

    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(__half)));

    CUDA_CHECK(cudaMemset(A, 0, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(B, 0, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(__half)));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i{}; i < warmup; ++i) {
        matmul_optimized_fp16<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());      // catch bad launch config
    CUDA_CHECK(cudaDeviceSynchronize()); // catch errors during execution

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for(int i{}; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_optimized_fp16<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    std::sort(times.begin(), times.end());
    double median = times[iters / 2] / 1000.0;

    return median;

}

SweepResults run_sweep_naive_fp32(const std::vector<int>& sizes) {
    SweepResults r;
    r.sizes = sizes;
    for (int N : sizes) {
        fprintf(stderr, "  fp32 N=%d...\n", N);
        MatmulStats s = matmul_arithmetic_intensity(N, 4);
        double elapsed = benchmark_matmul_fp32_naive(N);
        r.intensities.push_back(s.arithmetic_intensity);
        r.tflops.push_back((double)s.flops / elapsed / 1e12);
    }
    return r;
} 

SweepResults run_sweep_naive_fp16(const std::vector<int>& sizes) {
    SweepResults r;
    r.sizes = sizes;
    for (int N : sizes) {
        fprintf(stderr, "  fp16 N=%d...\n", N);
        MatmulStats s = matmul_arithmetic_intensity(N, 2);
        double elapsed = benchmark_matmul_fp16_naive(N);
        r.intensities.push_back(s.arithmetic_intensity);
        r.tflops.push_back((double)s.flops / elapsed / 1e12);
    }
    return r;
} 

SweepResults run_sweep_optimized_fp32(const std::vector<int>& sizes) {
    SweepResults r;
    r.sizes = sizes;
    for (int N : sizes) {
        fprintf(stderr, "  fp32 N=%d...\n", N);
        MatmulStats s = matmul_arithmetic_intensity(N, 4);
        double elapsed = benchmark_matmul_fp32_optimized(N);
        r.intensities.push_back(s.arithmetic_intensity);
        r.tflops.push_back((double)s.flops / elapsed / 1e12);
    }
    return r;
} 

SweepResults run_sweep_optimized_fp16(const std::vector<int>& sizes) {
    SweepResults r;
    r.sizes = sizes;
    for (int N : sizes) {
        fprintf(stderr, "  fp16 N=%d...\n", N);
        MatmulStats s = matmul_arithmetic_intensity(N, 2);
        double elapsed = benchmark_matmul_fp16_optimized(N);
        r.intensities.push_back(s.arithmetic_intensity);
        r.tflops.push_back((double)s.flops / elapsed / 1e12);
    }
    return r;
} 

int main() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "Device: %s\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    fprintf(stderr, "Benchmarking fp32...\n");
    SweepResults fp32_results = run_sweep_fp32(handle, sizes);

    fprintf(stderr, "Benchmarking fp16...\n");
    SweepResults fp16_results = run_sweep_fp16(handle, sizes);

    fprintf(stderr, "Benchmarking fp32 naive kernel...\n");
    SweepResults fp32_naive_results = run_sweep_naive_fp32(sizes);

    fprintf(stderr, "Benchmarking fp16 naive kernel...\n");
    SweepResults fp16_naive_results = run_sweep_naive_fp16(sizes);

    fprintf(stderr, "Benchmarking fp32 optimized kernel...\n");
    SweepResults fp32_optimized_results = run_sweep_optimized_fp32(sizes);

    fprintf(stderr, "Benchmarking fp16 optimized kernel...\n");
    SweepResults fp16_optimized_results = run_sweep_optimized_fp16(sizes);

    // Print CSV to stdout; redirect to roofline.csv and plot separately
    printf("dtype,kind,size_or_x,intensity_or_x,tflops\n");
    print_results("fp32", PEAK_FP32_TFLOPS, fp32_results);
    print_results("fp16", PEAK_FP16_TFLOPS, fp16_results);
    print_results("fp32_naive", PEAK_FP32_TFLOPS, fp32_naive_results);
    print_results("fp16_naive", PEAK_FP16_TFLOPS, fp16_naive_results);
    print_results("fp32_optimized", PEAK_FP32_TFLOPS, fp32_optimized_results);
    print_results("fp16_optimized", PEAK_FP16_TFLOPS, fp16_optimized_results);

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
