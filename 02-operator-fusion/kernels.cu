/*
 * Project B: CUDA operator fusion — benchmark fused kernels (ReLU, bias+GELU,
 * add+LayerNorm) vs unfused equivalents.
 *
 * Standalone benchmark:  make && ./kernels > results.csv && python3 plot_fusion.py results.csv
 * PyTorch extension:     python setup.py build_ext --inplace && python benchmark.py
 */

// torch/extension.h must come first when building the extension
#ifdef TORCH_EXTENSION
#include <torch/extension.h>
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

// RTX 5060 Laptop GPU (Blackwell GB206)
static constexpr double PEAK_BW_GB_S = 384.0;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));                 \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// returns median kernel time in milliseconds
template <typename Fn>
double benchmark_ms(Fn&& fn, int warmup = 5, int iters = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

// ===========================================================================
// Kernel 1: ReLU
//   y[i] = max(0, x[i])
//   Reads: N floats   Writes: N floats   Total bytes: 2 * N * 4
// ===========================================================================

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for(; i < N / 4; i+= stride) {

        float4 vals = x4[i];
        vals.x = fmaxf(0.0f, vals.x);
        vals.y = fmaxf(0.0f, vals.y);
        vals.z = fmaxf(0.0f, vals.z);
        vals.w = fmaxf(0.0f, vals.w);
        y4[i] = vals;
    }

    int remainder_start = (N / 4) * 4;
    int remainder_idx = remainder_start + (blockIdx.x * blockDim.x + threadIdx.x);

    if (remainder_idx < N) {
        y[remainder_idx] = fmaxf(0.0f, x[remainder_idx]);
    }
}

void run_relu(int N) {
    float *device_input, *device_output;
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));

    std::vector<float> host_input(N);
    for (int i = 0; i < N; ++i) host_input[i] = static_cast<float>(i % 7) - 3.0f;
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int vec_work = (N + 3) / 4;
    int blocks = std::min((vec_work + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;

    relu_kernel<<<blocks, threads>>>(device_input, device_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> host_output(N);
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        float expected = fmaxf(0.0f, host_input[i]);
        if (host_output[i] != expected) {
            fprintf(stderr, "relu mismatch at %d: got %f want %f\n", i, host_output[i], expected);
            exit(1);
        }
    }

    double ms = benchmark_ms([&]() {relu_kernel<<<blocks, threads>>>(device_input, device_output, N);});

    double moved_bytes = 2.0 * static_cast<double>(bytes);
    double bw_gb_s = moved_bytes / (ms * 1e-3) / 1e9;

    printf("relu,fused,%d,0,%.5f,%.2f\n", N, ms, bw_gb_s);
    fprintf(stderr, "  N=%8d  %.5f ms  %.1f GB/s  (%.0f%% of peak)\n", N, ms, bw_gb_s, 100.0 * bw_gb_s / PEAK_BW_GB_S);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
}

// ===========================================================================
// Kernel 2a (unfused): Bias add then GELU — two separate kernels
//   pass 1:  y[i] = x[i] + bias[i % C]
//   pass 2:  y[i] = gelu(y[i])
// ===========================================================================

//possibly put bias into shared memory can have each of the threads read 1 of the bias elements before then sync, but only works if C < 64kb?
__global__ void bias_add_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for(; i < N /4; i += stride) {
        float4 vals = x4[i];

        int raw_idx = i * 4;

        float bias_x = bias[raw_idx % C];
        float bias_y = bias[(raw_idx + 1) % C];
        float bias_z = bias[(raw_idx + 2) % C];
        float bias_w = bias[(raw_idx + 3) % C];

        float4 result;
        result.x = vals.x + bias_x;
        result.y = vals.y + bias_y;
        result.z = vals.z + bias_z;
        result.w = vals.w + bias_w;

        y4[i] = result;
    }

    int remainder_start = (N / 4) * 4;
    int remainder_idx = remainder_start + blockIdx.x * blockDim.x + threadIdx.x;

    if (remainder_idx < N) {
        y[remainder_idx] = x[remainder_idx] + bias[remainder_idx % C];
    }
}

__global__ void gelu_kernel(float* __restrict__ y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float4* y4 = reinterpret_cast<float4*>(y);

    for(; i < N / 4; i += stride) {
        float4 vals = y4[i];
        float4 result;

        result.x = vals.x * 0.5f * (1.0f + erff(vals.x / sqrtf(2)));
        result.y = vals.y * 0.5f * (1.0f + erff(vals.y / sqrtf(2)));
        result.z = vals.z * 0.5f * (1.0f + erff(vals.z / sqrtf(2)));
        result.w = vals.w * 0.5f * (1.0f + erff(vals.w / sqrtf(2)));

        y4[i] = result;
    }

    int remainder_start = (N / 4) * 4;
    int remainder_idx = remainder_start + blockDim.x * blockIdx.x + threadIdx.x;

    if (remainder_idx < N) {
        y[remainder_idx] = y[remainder_idx] * 0.5f * (1.0f + erff(y[remainder_idx] / sqrtf(2)));
    }
}

// ===========================================================================
// Kernel 2b (fused): Bias + GELU — single kernel, one global mem read/write
// ===========================================================================

__global__ void bias_gelu_fused_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for(; i < N / 4; i += stride) {
        float4 vals = x4[i];
        float4 result;
        int raw_idx = i * 4;

        result.x = vals.x + bias[raw_idx % C];
        result.y = vals.y + bias[(raw_idx + 1) % C];
        result.z = vals.z + bias[(raw_idx + 2) % C];
        result.w = vals.w + bias[(raw_idx + 3) % C];

        result.x = result.x * 0.5f * (1.0f + erff(result.x / sqrtf(2)));
        result.y = result.y * 0.5f * (1.0f + erff(result.y / sqrtf(2)));
        result.z = result.z * 0.5f * (1.0f + erff(result.z / sqrtf(2)));
        result.w = result.w * 0.5f * (1.0f + erff(result.w / sqrtf(2)));

        y4[i] = result;
    }

    int remainder_start = (N / 4) * 4;
    int remainder_idx = remainder_start + blockDim.x * blockIdx.x + threadIdx.x;

    if (remainder_idx < N) {
        float biased = x[remainder_idx] + bias[remainder_idx % C];
        y[remainder_idx] = biased * 0.5f * (1.0f + erff(biased / sqrtf(2)));
    }
}

void run_bias_gelu(int N, int C) {
    float *device_input, *device_bias, *device_output;

    size_t input_bytes = static_cast<size_t>(N) * sizeof(float);
    size_t bias_bytes = static_cast<size_t>(C) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&device_bias, bias_bytes));
    CUDA_CHECK(cudaMalloc(&device_output, input_bytes));

    std::vector<float> host_input(N);
    std::vector<float> host_bias(C);

    for (int i = 0; i < N; ++i) host_input[i] = static_cast<float>(i % 7) - 3.0f;
    for (int i = 0; i < C; ++i) host_bias[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;

    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_bias, host_bias.data(), bias_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int vec_work = (N + 3) / 4;
    int blocks = std::min((vec_work + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;

    // check correctness: fused kernel vs CPU reference gelu(x + bias)
    bias_gelu_fused_kernel<<<blocks, threads>>>(device_input, device_bias, device_output, N, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> host_output(N);
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, input_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        float v = host_input[i] + host_bias[i % C];
        float expected = v * 0.5f * (1.0f + erff(v / sqrtf(2.0f)));
        if (fabsf(host_output[i] - expected) > 1e-3f) {
            fprintf(stderr, "bias_gelu mismatch at %d: got %f want %f\n", i, host_output[i], expected);
            exit(1);
        }
    }

    // ideal traffic (read x, write y, read bias once) is charged the same to both
    // variants; the unfused path's extra round trip shows up as lower effective BW.
    double ideal_bytes = static_cast<double>(2.0 * N + C) * sizeof(float);

    // unfused: bias add, then a separate pass for gelu
    double ms_unfused = benchmark_ms([&]() {
        bias_add_kernel<<<blocks, threads>>>(device_input, device_bias, device_output, N, C);
        gelu_kernel<<<blocks, threads>>>(device_output, N);
    });
    double bw_unfused = ideal_bytes / (ms_unfused * 1e-3) / 1e9;
    printf("bias_gelu,unfused,%d,%d,%.5f,%.2f\n", N, C, ms_unfused, bw_unfused);

    // fused: single pass
    double ms_fused = benchmark_ms([&]() {
        bias_gelu_fused_kernel<<<blocks, threads>>>(device_input, device_bias, device_output, N, C);
    });
    double bw_fused = ideal_bytes / (ms_fused * 1e-3) / 1e9;
    printf("bias_gelu,fused,%d,%d,%.5f,%.2f\n", N, C, ms_fused, bw_fused);

    fprintf(stderr, "  N=%8d  unfused %.5f ms (%.1f GB/s)  fused %.5f ms (%.1f GB/s)\n",
            N, ms_unfused, bw_unfused, ms_fused, bw_fused);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_bias));
    CUDA_CHECK(cudaFree(device_output));
}

// ===========================================================================
// Kernel 3a (unfused): Residual add then LayerNorm — two separate kernels
//   pass 1:  out[i] = x[i] + residual[i]
//   pass 2:  out = layernorm(out, gamma, beta)   (over last dim C)
//
//   LayerNorm: for each row r of length C:
//     mean  = sum(row) / C
//     var   = sum((row - mean)^2) / C
//     out[r,c] = gamma[c] * (row[c] - mean) / sqrt(var + eps) + beta[c]
// ===========================================================================

__device__ __forceinline__ double warp_reduce_sum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void add_kernel(const float* __restrict__ x, const float* __restrict__ residual, float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* r4 = reinterpret_cast<const float4*>(residual);
    float4* out4 = reinterpret_cast<float4*>(out);

    for(; i < N / 4; i +=stride) {
        float4 vals = x4[i];
        float4 res_vals = r4[i];
        float4 result;

        result.x = vals.x + res_vals.x;
        result.y = vals.y + res_vals.y;
        result.z = vals.z + res_vals.z;
        result.w = vals.w + res_vals.w;

        out4[i] = result;
    }

    int remainder_start = (N / 4) * 4;
    int remainder_idx = remainder_start + blockIdx.x * blockDim.x + threadIdx.x;

    if (remainder_idx < N) {
        out[remainder_idx] = x[remainder_idx] + residual[remainder_idx];
    }
}

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, int rows, int C, float eps) {

    int r = blockIdx.x;

    if (r >= rows) {
        return;
    }

    int tid = threadIdx.x;

    double thread_sum{};
    double thread_sq_sum{};

    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (int col = tid; col < C / 4; col += blockDim.x) {
        int global_idx = r * C / 4 + col;

        float4 vals = x4[global_idx];

        double v1 = static_cast<double>(vals.x);
        double v2 = static_cast<double>(vals.y);
        double v3 = static_cast<double>(vals.z);
        double v4 = static_cast<double>(vals.w);

        thread_sum += v1 + v2 + v3 + v4;
        thread_sq_sum += (v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4);
    }

    int remainder_start = (C / 4) * 4;
    int remainder_idx = remainder_start + tid;

    if (remainder_idx < C) {
        int global_idx = r * C + remainder_idx;
        double val = static_cast<double>(x[global_idx]);
        thread_sum += val;
        thread_sq_sum += val * val;
    }

    thread_sum = warp_reduce_sum(thread_sum);
    thread_sq_sum = warp_reduce_sum(thread_sq_sum);

    __shared__ double shared_sum[32];
    __shared__ double shared_sq_sum[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
        shared_sq_sum[warp_id] = thread_sq_sum;
    }

    __syncthreads();

    __shared__ double final_mean;
    __shared__ double final_var;

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        double block_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0;
        double block_sq_sum = (lane_id < num_warps) ? shared_sq_sum[lane_id] : 0.0;

        block_sum = warp_reduce_sum(block_sum);
        block_sq_sum = warp_reduce_sum(block_sq_sum);

        if (lane_id == 0) {
            final_mean = block_sum / C;
            final_var = (block_sq_sum / C) - (final_mean * final_mean);
            if (final_var < 0.0) {
                final_var = 0.0;
            }
        }
    }

    __syncthreads();

    float4* out4 = reinterpret_cast<float4*>(out);

    float mean_f = static_cast<float>(final_mean);
    float var_f = static_cast<float>(final_var);
    float rsqrt_std = rsqrtf(var_f + eps);

    for(int col = tid; col < C / 4; col += blockDim.x) {

        int global_idx = r * C / 4 + col;
        float4 result;
        float4 vals = x4[global_idx];
        int scalar_col = col * 4;
        result.x = gamma[scalar_col] * (vals.x - mean_f) * rsqrt_std + beta[scalar_col];
        result.y = gamma[scalar_col + 1] * (vals.y - mean_f) * rsqrt_std + beta[scalar_col+ 1];
        result.z = gamma[scalar_col+ 2] * (vals.z - mean_f) * rsqrt_std + beta[scalar_col + 2];
        result.w = gamma[scalar_col + 3] * (vals.w - mean_f) * rsqrt_std + beta[scalar_col + 3];

        out4[global_idx] = result;
    }

    if (remainder_idx < C) {
        int global_idx = r * C + remainder_idx;
        out[global_idx] = gamma[remainder_idx] * (x[global_idx] - mean_f) / sqrtf(var_f + eps) + beta[remainder_idx];
    }
}

// ===========================================================================
// Kernel 3b (fused): Add + LayerNorm — single kernel
// ===========================================================================

__global__ void add_layernorm_fused_kernel(const float* __restrict__ x, const float* __restrict__ residual, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, int rows, int C, float eps) {
    int r = blockIdx.x;
    int tid = threadIdx.x;

    double thread_sum{};
    double thread_sq_sum{};

    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* r4 = reinterpret_cast<const float4*>(residual);

    for(int col = tid; col < C / 4; col += blockDim.x) {
        int global_idx = r * C / 4 + col;

        float4 vals = x4[global_idx];
        float4 res_vals = r4[global_idx];

        double v1 = static_cast<double>(vals.x + res_vals.x);
        double v2 = static_cast<double>(vals.y + res_vals.y);
        double v3 = static_cast<double>(vals.z + res_vals.z);
        double v4 = static_cast<double>(vals.w + res_vals.w);

        thread_sum += v1 + v2 + v3 + v4;
        thread_sq_sum += v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4;
    }

    int remainder_start = (C / 4) * 4;
    int remainder_idx = remainder_start + tid;

    if (remainder_idx < C) {
        int global_idx = r * C + remainder_idx;
        double val = static_cast<double>(x[global_idx] + residual[global_idx]);
        thread_sum += val;
        thread_sq_sum += val * val;
    }

    thread_sum = warp_reduce_sum(thread_sum);
    thread_sq_sum = warp_reduce_sum(thread_sq_sum);

    __shared__ double shared_sum[32];
    __shared__ double shared_sq_sum[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
        shared_sq_sum[warp_id] = thread_sq_sum;
    }

    __syncthreads();

    __shared__ double final_mean;
    __shared__ double final_var;

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        double block_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0;
        double block_sq_sum = (lane_id < num_warps) ? shared_sq_sum[lane_id] : 0.0;

        block_sum = warp_reduce_sum(block_sum);
        block_sq_sum = warp_reduce_sum(block_sq_sum);

        if (lane_id == 0) {
            final_mean = block_sum / C;
            final_var = block_sq_sum / C - (final_mean * final_mean);
            if (final_var < 0.0) {
                final_var = 0.0;
            }
        }
    }

    __syncthreads();

    float mean_f = static_cast<float>(final_mean);
    float var_f = static_cast<float>(final_var);
    float4* out4 = reinterpret_cast<float4*>(out);

    float rsqrt_std = rsqrtf(var_f + eps);

    for(int col = tid; col < C / 4; col += blockDim.x) {
        int global_idx = r * C / 4 + col;
        int scalar_col = col * 4;
        float4 vals = x4[global_idx];
        float4 res_vals = r4[global_idx];
        float4 result;

        result.x = gamma[scalar_col] * (vals.x + res_vals.x - mean_f) * rsqrt_std + beta[scalar_col];
        result.y = gamma[scalar_col + 1] * (vals.y + res_vals.y - mean_f) * rsqrt_std + beta[scalar_col + 1];
        result.z = gamma[scalar_col + 2] * (vals.z + res_vals.z - mean_f) * rsqrt_std + beta[scalar_col + 2];
        result.w = gamma[scalar_col + 3] * (vals.w + res_vals.w - mean_f) * rsqrt_std + beta[scalar_col + 3];

        out4[global_idx] = result;
    }

    if(remainder_idx < C) {
        int global_idx = r * C + remainder_idx;
        out[global_idx] = gamma[remainder_idx] * (x[global_idx] + residual[global_idx] - mean_f) / sqrtf(var_f + eps) + beta[remainder_idx];
    }
}

void run_add_layernorm(int rows, int C) {
    int N = rows * C;
    float *device_input, *device_residual, *device_gamma, *device_beta, *device_tmp, *device_output;
    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    size_t param_bytes = static_cast<size_t>(C) * sizeof(float);
    float eps = 1e-5f;

    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_residual, bytes));
    CUDA_CHECK(cudaMalloc(&device_gamma, param_bytes));
    CUDA_CHECK(cudaMalloc(&device_beta, param_bytes));
    // device_tmp holds the intermediate for the unfused path
    CUDA_CHECK(cudaMalloc(&device_tmp, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));

    std::vector<float> host_input(N), host_residual(N);
    for (int i = 0; i < N; ++i) {
        host_input[i]    = static_cast<float>(i % 13) * 0.1f - 0.6f;
        host_residual[i] = static_cast<float>(i % 7)  * 0.1f - 0.3f;
    }
    std::vector<float> host_gamma(C), host_beta(C);
    for (int i = 0; i < C; ++i) {
        host_gamma[i] = 1.0f;
        host_beta[i]  = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(device_input,    host_input.data(),    bytes,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_residual, host_residual.data(), bytes,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_gamma,    host_gamma.data(),    param_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_beta,     host_beta.data(),     param_bytes, cudaMemcpyHostToDevice));

    // layernorm kernels launch one block per row; the add kernel is a flat grid-stride loop
    int threads = 256;
    int add_vec_work = (N + 3) / 4;
    int add_blocks = std::min((add_vec_work + threads - 1) / threads, 1024);
    if (add_blocks < 1) add_blocks = 1;

    // correctness: fused kernel vs CPU reference layernorm(x + residual)
    add_layernorm_fused_kernel<<<rows, threads>>>(device_input, device_residual, device_gamma, device_beta, device_output, rows, C, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> host_output(N);
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        double mean = 0.0, sq = 0.0;
        for (int c = 0; c < C; ++c) {
            double v = static_cast<double>(host_input[r * C + c]) + host_residual[r * C + c];
            mean += v;
            sq   += v * v;
        }
        mean /= C;
        double var = sq / C - mean * mean;
        if (var < 0.0) var = 0.0;
        double inv_std = 1.0 / sqrt(var + eps);
        for (int c = 0; c < C; ++c) {
            double v = static_cast<double>(host_input[r * C + c]) + host_residual[r * C + c];
            float expected = static_cast<float>(host_gamma[c] * (v - mean) * inv_std + host_beta[c]);
            if (fabsf(host_output[r * C + c] - expected) > 1e-2f) {
                fprintf(stderr, "add_layernorm mismatch at row %d col %d: got %f want %f\n",
                        r, c, host_output[r * C + c], expected);
                exit(1);
            }
        }
    }

    // ideal traffic: read x + residual, write out, read gamma + beta once
    double ideal_bytes = static_cast<double>(3.0 * N + 2.0 * C) * sizeof(float);

    // unfused: residual add into a temp buffer, then a separate layernorm pass
    double ms_unfused = benchmark_ms([&]() {
        add_kernel<<<add_blocks, threads>>>(device_input, device_residual, device_tmp, N);
        layernorm_kernel<<<rows, threads>>>(device_tmp, device_gamma, device_beta, device_output, rows, C, eps);
    });
    double bw_unfused = ideal_bytes / (ms_unfused * 1e-3) / 1e9;
    printf("add_layernorm,unfused,%d,%d,%.5f,%.2f\n", N, C, ms_unfused, bw_unfused);

    // fused: single pass
    double ms_fused = benchmark_ms([&]() {
        add_layernorm_fused_kernel<<<rows, threads>>>(device_input, device_residual, device_gamma, device_beta, device_output, rows, C, eps);
    });
    double bw_fused = ideal_bytes / (ms_fused * 1e-3) / 1e9;
    printf("add_layernorm,fused,%d,%d,%.5f,%.2f\n", N, C, ms_fused, bw_fused);

    fprintf(stderr, "  rows=%6d C=%4d  unfused %.5f ms (%.1f GB/s)  fused %.5f ms (%.1f GB/s)\n",
            rows, C, ms_unfused, bw_unfused, ms_fused, bw_fused);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_residual));
    CUDA_CHECK(cudaFree(device_gamma));
    CUDA_CHECK(cudaFree(device_beta));
    CUDA_CHECK(cudaFree(device_tmp));
    CUDA_CHECK(cudaFree(device_output));
}

// ===========================================================================
// SECTION C: PyTorch C++ extension — built by setup.py (defines
// TORCH_EXTENSION), exposes the kernels to Python via pybind11
// ===========================================================================
#ifdef TORCH_EXTENSION

at::Tensor relu_fwd(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    x = x.contiguous();

    auto y = at::empty_like(x);
    int N = x.numel();

    int threads = 256;
    int vec_work = (N + 3) / 4;
    int blocks = std::min((vec_work + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;

    relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    CUDA_CHECK(cudaGetLastError());

    return y;
}

at::Tensor gelu_fwd(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");

    // gelu_kernel runs in place, so transform a contiguous copy of x
    auto y = x.contiguous().clone();
    int N = y.numel();

    int threads = 256;
    int vec_work = (N + 3) / 4;
    int blocks = std::min((vec_work + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;

    gelu_kernel<<<blocks, threads>>>(y.data_ptr<float>(), N);
    CUDA_CHECK(cudaGetLastError());

    return y;
}

// x (N,), bias (C,) where N is divisible by C
at::Tensor bias_gelu_fwd(at::Tensor x, at::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "x and bias must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    x = x.contiguous();
    bias = bias.contiguous();

    auto y = at::empty_like(x);
    int N = x.numel();
    int C = bias.numel();

    int threads = 256;
    int vec_work = (N + 3) / 4;
    int blocks = std::min((vec_work + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;

    bias_gelu_fused_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), N, C);
    CUDA_CHECK(cudaGetLastError());

    return y;
}

// x (rows, C), residual (rows, C), gamma (C,), beta (C,)
at::Tensor add_layernorm_fwd(at::Tensor x, at::Tensor residual, at::Tensor gamma, at::Tensor beta, float eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "inputs must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (rows, C)");
    x = x.contiguous();
    residual = residual.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    auto y = at::empty_like(x);
    int rows = x.size(0);
    int C = x.size(1);

    // one block per row: the layernorm reduction is per-row
    int threads = 256;
    add_layernorm_fused_kernel<<<rows, threads>>>(x.data_ptr<float>(), residual.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), rows, C, eps);
    CUDA_CHECK(cudaGetLastError());

    return y;
}

// TORCH_EXTENSION_NAME is set by setup.py; in Python: import activations_cuda
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Project B: fused activation CUDA kernels";

    m.def("relu_fwd", &relu_fwd, "ReLU forward");
    m.def("gelu_fwd", &gelu_fwd, "GELU forward (unfused)");
    m.def("bias_gelu_fwd", &bias_gelu_fwd, "Bias + GELU fused forward");
    m.def("add_layernorm_fwd", &add_layernorm_fwd, "Add + LayerNorm fused forward");
}

#endif


// ===========================================================================
// SECTION B: standalone entry point (compiled by make, not setup.py)
// ===========================================================================
#ifndef TORCH_EXTENSION

int main() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "Device: %s\n", prop.name);

    printf("kernel,variant,N,C,time_ms,bw_gb_s\n");

    std::vector<int> sizes = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};

    fprintf(stderr, "Kernel 1: ReLU...\n");
    for (int N : sizes) run_relu(N);

    fprintf(stderr, "Kernel 2: Bias + GELU...\n");
    int C = 1024;
    for (int N : sizes) run_bias_gelu(N, C);

    fprintf(stderr, "Kernel 3: Add + LayerNorm...\n");
    for (int N : sizes) run_add_layernorm(N / C, C);

    return 0;
}

#endif // !TORCH_EXTENSION
