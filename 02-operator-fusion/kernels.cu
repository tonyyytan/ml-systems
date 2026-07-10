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

static constexpr double PEAK_BW_GB_S = 272.0;   // RTX 5060 Laptop GPU (Blackwell GB206)

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
    // TODO: allocate, benchmark unfused vs fused, print CSV rows, free
    (void)device_input; (void)device_bias; (void)device_output;
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
    float *device_input, *device_residual, *device_gamma, *device_beta, *device_output;
    // TODO: allocate, benchmark unfused vs fused, print CSV rows, free
    (void)device_input; (void)device_residual; (void)device_gamma; (void)device_beta; (void)device_output;
}

// ===========================================================================
// SECTION C: PyTorch C++ extension — built by setup.py (defines
// TORCH_EXTENSION), exposes the kernels to Python via pybind11
// ===========================================================================
#ifdef TORCH_EXTENSION

at::Tensor relu_fwd(at::Tensor x) {
    // TODO: TORCH_CHECK(x.is_cuda()), x.contiguous(), y = at::empty_like(x),
    //       launch relu_kernel on x.data_ptr<float>(), return y
    return x;
}

at::Tensor gelu_fwd(at::Tensor x) {
    // TODO: same structure as relu_fwd but calling gelu_kernel
    return x;
}

// x (N,), bias (C,) where N is divisible by C
at::Tensor bias_gelu_fwd(at::Tensor x, at::Tensor bias) {
    // TODO
    return x;
}

// x (rows, C), residual (rows, C), gamma (C,), beta (C,)
at::Tensor add_layernorm_fwd(at::Tensor x, at::Tensor residual,
                              at::Tensor gamma, at::Tensor beta,
                              float eps) {
    // TODO
    return x;
}

// TORCH_EXTENSION_NAME is set by setup.py; in Python: import activations_cuda
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Project B: fused activation CUDA kernels";

    m.def("relu_fwd",          &relu_fwd,          "ReLU forward");
    m.def("gelu_fwd",          &gelu_fwd,          "GELU forward (unfused)");

    // TODO: uncomment once implemented
    // m.def("bias_gelu_fwd",     &bias_gelu_fwd,     "Bias + GELU fused forward");
    // m.def("add_layernorm_fwd", &add_layernorm_fwd, "Add + LayerNorm fused forward");
}

#endif // TORCH_EXTENSION


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
