/*
 * Project B: CUDA operator fusion — benchmark three fused kernels vs unfused
 *            PyTorch equivalents to understand the memory bandwidth savings
 *            of operator fusion.
 *
 * Kernels:
 *   1. ReLU            — baseline elementwise, trivially memory-bound
 *   2. Bias + GELU     — fuse bias add + GELU activation in one pass
 *   3. Add + LayerNorm — fuse residual add + layer norm (transformer block)
 *
 * For each kernel we compare:
 *   (a) unfused: separate CUDA kernels / PyTorch ops, each touching global mem
 *   (b) fused:   single kernel that reads once, writes once
 *
 * Build (standalone C++ benchmark):
 *   make
 *   ./kernels > results.csv
 *   python3 plot_fusion.py results.csv
 *
 * Build (PyTorch extension for benchmark.py):
 *   python setup.py build_ext --inplace
 *   python benchmark.py
 *
 * The file is split into three sections:
 *   A) Raw CUDA kernels          — __global__ functions, no PyTorch dependency
 *   B) Standalone C++ main()     — compiled by make, uses cudaMalloc directly
 *   C) PyTorch extension glue    — compiled by setup.py, exposes ops to Python
 *      (guarded by #ifdef TORCH_EXTENSION — set automatically by setup.py)
 */

// ---------------------------------------------------------------------------
// Includes
// ---------------------------------------------------------------------------
// When building the PyTorch extension, torch/extension.h must come first.
// The #ifdef lets the same file compile both as a standalone binary (make)
// and as a Python extension (setup.py).
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

// ---------------------------------------------------------------------------
// Hardware constants — RTX 5060 Laptop GPU (Blackwell GB206)
// ---------------------------------------------------------------------------
static constexpr double PEAK_BW_GB_S = 272.0;   // memory bandwidth GB/s

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

// ---------------------------------------------------------------------------
// Benchmark helper — returns median kernel time in milliseconds
// ---------------------------------------------------------------------------
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

__global__ void relu_kernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int N) {
    // TODO: implement
    // hint: grid-stride loop over elements
}

void run_relu(int N) {
    float *d_x, *d_y;
    // TODO: allocate, fill, benchmark, print CSV row, free
    (void)d_x; (void)d_y;
}

// ===========================================================================
// Kernel 2a (unfused): Bias add then GELU — two separate kernels
//   pass 1:  y[i] = x[i] + bias[i % C]
//   pass 2:  y[i] = gelu(y[i])
// ===========================================================================

__global__ void bias_add_kernel(const float* __restrict__ x,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y,
                                 int N, int C) {
    // TODO: implement
}

__global__ void gelu_kernel(float* __restrict__ y, int N) {
    // TODO: implement
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
}

// ===========================================================================
// Kernel 2b (fused): Bias + GELU — single kernel, one global mem read/write
// ===========================================================================

__global__ void bias_gelu_fused_kernel(const float* __restrict__ x,
                                        const float* __restrict__ bias,
                                        float* __restrict__ y,
                                        int N, int C) {
    // TODO: implement — combine bias add and gelu in one pass
}

void run_bias_gelu(int N, int C) {
    float *d_x, *d_bias, *d_y;
    // TODO: allocate, benchmark unfused vs fused, print CSV rows, free
    (void)d_x; (void)d_bias; (void)d_y;
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

__global__ void add_kernel(const float* __restrict__ x,
                            const float* __restrict__ residual,
                            float* __restrict__ out,
                            int N) {
    // TODO: implement
}

__global__ void layernorm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ out,
                                  int rows, int C,
                                  float eps) {
    // TODO: implement — one block per row, use shared memory for reduction
    // hint: each block handles one row; threads cooperate to compute mean/var
}

// ===========================================================================
// Kernel 3b (fused): Add + LayerNorm — single kernel
// ===========================================================================

__global__ void add_layernorm_fused_kernel(const float* __restrict__ x,
                                            const float* __restrict__ residual,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            float* __restrict__ out,
                                            int rows, int C,
                                            float eps) {
    // TODO: implement — add residual and normalize in one pass
}

void run_add_layernorm(int rows, int C) {
    float *d_x, *d_residual, *d_gamma, *d_beta, *d_out;
    // TODO: allocate, benchmark unfused vs fused, print CSV rows, free
    (void)d_x; (void)d_residual; (void)d_gamma; (void)d_beta; (void)d_out;
}

// ===========================================================================
// SECTION C: PyTorch C++ Extension
//
// Only compiled when building via setup.py (which defines TORCH_EXTENSION).
// This section exposes the raw CUDA kernels above to Python via pybind11.
//
// Key concepts:
//
//   at::Tensor          — the PyTorch tensor type in C++
//   tensor.data_ptr<scalar_t>()   — raw pointer to the underlying data
//   tensor.numel()      — total number of elements (like .numel() in Python)
//   tensor.contiguous() — ensures elements are laid out without gaps/strides
//   at::empty_like(x)   — allocate output with same shape/dtype/device as x
//
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "name", [&]() {
//       using scalar_t = ...;  // float or at::Half, resolved at runtime
//       // launch kernel using scalar_t pointers
//   });
//
//   Kernel launch syntax (reminder):
//       int threads = 256;
//       int blocks  = (N + threads - 1) / threads;   // ceil(N / threads)
//       my_kernel<<<blocks, threads>>>(args...);
//
//   TORCH_CHECK(condition, "error message")  — like assert but for PyTorch ops
// ===========================================================================
#ifdef TORCH_EXTENSION

// --------------------------------------------------------------------------
// relu_fwd: wraps relu_kernel
// --------------------------------------------------------------------------
at::Tensor relu_fwd(at::Tensor x) {
    // TODO:
    //   1. TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor")
    //   2. x = x.contiguous()
    //   3. auto y = at::empty_like(x)
    //   4. int N = x.numel()
    //   5. int threads = 256, blocks = (N + threads - 1) / threads
    //   6. AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "relu_fwd", [&]() {
    //          relu_kernel<<<blocks, threads>>>(
    //              x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), N);
    //      });
    //   7. return y
    return x; // placeholder — remove once implemented
}

// --------------------------------------------------------------------------
// TODO: gelu_fwd — wraps gelu_kernel (unfused, no bias)
// --------------------------------------------------------------------------
at::Tensor gelu_fwd(at::Tensor x) {
    // TODO: same structure as relu_fwd but calling gelu_kernel
    return x;
}

// --------------------------------------------------------------------------
// TODO: bias_gelu_fwd — wraps bias_gelu_fused_kernel
//   Takes x (N,) and bias (C,) where N is divisible by C
// --------------------------------------------------------------------------
at::Tensor bias_gelu_fwd(at::Tensor x, at::Tensor bias) {
    // TODO
    return x;
}

// --------------------------------------------------------------------------
// TODO: add_layernorm_fwd — wraps add_layernorm_fused_kernel
//   Takes x (rows, C), residual (rows, C), gamma (C,), beta (C,), eps
// --------------------------------------------------------------------------
at::Tensor add_layernorm_fwd(at::Tensor x, at::Tensor residual,
                              at::Tensor gamma, at::Tensor beta,
                              float eps) {
    // TODO
    return x;
}

// --------------------------------------------------------------------------
// Module registration
//
// PYBIND11_MODULE(name, m) { m.def(...) } is the pybind11 way to expose
// C++ functions to Python. TORCH_EXTENSION_NAME is filled in by setup.py.
//
// After building, in Python:
//   import activations_cuda as A
//   y = A.relu_fwd(x)
// --------------------------------------------------------------------------
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
// SECTION B: Standalone C++ entry point (compiled by make, not setup.py)
// ===========================================================================
#ifndef TORCH_EXTENSION

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "Device: %s\n", prop.name);

    // Print CSV header
    printf("kernel,variant,N,C,time_ms,bw_gb_s\n");

    // Sweep sizes
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
