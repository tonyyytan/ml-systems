/*
 * Project 03, Step 4: quantized GEMV.
 *
 * GEMV = GEneral Matrix-Vector multiply:  y = W @ x
 *
 *     W : (M, K) fp16, row-major   the weight matrix (BIG -- this is the traffic)
 *     x : (K,)   fp16              one token's activation vector (small)
 *     y : (M,)   fp16              the output
 *
 * This IS decode at batch 1. Every weight is read exactly once and used for
 * exactly one multiply-add: 2 flops per 2 bytes = 1 flop/byte. The GPU can do
 * hundreds of flops per byte delivered, so it starves. The kernel is bound by
 * memory bandwidth (272 GB/s on this card), NOT by arithmetic.
 *
 * Consequence: the ONLY figure of merit here is achieved bandwidth.
 *     GB/s = bytes_read / seconds        bytes_read = M * K * sizeof(weight)
 * A "faster" kernel that reads the same bytes cannot help. Reading FEWER bytes
 * (quantization) is the whole game -- but that comes after this file works.
 *
 * Build order in this file, each one a separate measurement:
 *   1. gemv_fp16   -- baseline. no quantization. exists to give an honest
 *                     number to beat and to prove we can approach 272 GB/s.
 *   2. gemv_w8a16  -- int8 weights, fp16 activations, dequant FUSED IN-REGISTER.
 *   3. gemv_w8a16_fp8 -- same bytes as int8, fp8 e4m3 encoding instead.
 *
 * Only #1 is scaffolded here. Do not start #2 until #1 hits ~85%+ of peak;
 * a quantization speedup measured against a bad baseline means nothing.
 *
 * Build:  cd 03-inference-engine && python3 setup.py build_ext --inplace
 */

// torch/extension.h must come first when building as an extension.
// These three are torch-only on purpose: they stay inside the ifdef so the
// standalone `make` build (which does NOT define TORCH_EXTENSION) never needs
// libtorch headers.
#ifdef TORCH_EXTENSION
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK()
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// RTX 5060 Laptop GPU (Blackwell GB206). The ceiling everything is measured against.
static constexpr double PEAK_BW_GB_S = 272.0;


/* ---------------------------------------------------------------------------
 * 1. fp16 baseline
 * -------------------------------------------------------------------------*/

/*
 * Suggested parallelisation -- ONE THREAD BLOCK PER OUTPUT ROW.
 *
 *   grid  = M blocks      (block m computes y[m], a single scalar)
 *   block = 256 threads   (the 256 threads cooperate on row m's K elements)
 *
 * So each block computes one dot product:  y[m] = sum over k of W[m][k] * x[k]
 *
 * Two things decide whether this hits peak bandwidth:
 *
 * (a) COALESCING. The GPU reads memory in wide contiguous chunks. If thread 0
 *     reads W[m][0], thread 1 reads W[m][1], thread 2 reads W[m][2] ... those
 *     land in one transaction and you get full bandwidth. If threads read
 *     scattered addresses, each one costs a separate transaction and you get a
 *     fraction of peak. So have consecutive threads read consecutive k, and
 *     stride the loop by blockDim.x -- NOT give each thread a contiguous chunk.
 *
 * (b) BYTES PER LOAD INSTRUCTION. One `half` is 2 bytes. Issuing a load
 *     instruction per 2 bytes usually cannot saturate the bus. Loading 16 bytes
 *     at a time (reinterpret the row pointer as float4 / half2) is what
 *     typically takes this kernel from ~50% to ~90% of peak.
 *     Get it CORRECT with scalar `half` loads first, measure, THEN vectorize
 *     and measure again. That delta is a result worth putting in the README.
 *
 * Accumulate in float, not half. Summing thousands of fp16 values in fp16
 * loses precision badly. The inputs are 2 bytes; the accumulator is a register.
 */
__global__ void gemv_fp16_kernel(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y, int M, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warps_per_block = blockDim.x / 32;

    extern __shared__ half shared_x[];

    for (int j{tid}; j < K; j += blockDim.x) {
        shared_x[j] = x[j];
    }
    __syncthreads();

    int row = blockIdx.x * warps_per_block + warp_id;

    if (row < M) {

        float thread_sum{};

        for (int j{lane_id}; j < K; j += 32) {
            thread_sum += __half2float(W[row * K + j]) * __half2float(shared_x[j]);
        }

        //warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        if (lane_id == 0) {
            y[row] = __float2half(thread_sum);
        }
    }
}


/*
 * Same mapping as above, but 16 bytes per load instruction instead of 2.
 * A float4 is 8 halves, so one instruction covers what took eight before and
 * the warp still walks the row contiguously (lane l takes float4 l, l+32, ...).
 *
 * Needs K % 8 == 0 and a 16-byte-aligned row start; K % 8 == 0 gives both,
 * since a row is K * 2 bytes and torch allocations are 512-byte aligned.
 */
__global__ void gemv_fp16_vec_kernel(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y, int M, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warps_per_block = blockDim.x / 32;

    extern __shared__ __align__(16) half shared_x[];

    const int k4 = K / 8;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* sx4 = reinterpret_cast<float4*>(shared_x);

    for (int j{tid}; j < k4; j += blockDim.x) {
        sx4[j] = x4[j];
    }
    __syncthreads();

    int row = blockIdx.x * warps_per_block + warp_id;

    if (row < M) {

        const float4* w4 = reinterpret_cast<const float4*>(W + static_cast<size_t>(row) * K);

        float thread_sum{};

        for (int j{lane_id}; j < k4; j += 32) {
            float4 wv = w4[j];
            float4 xv = sx4[j];

            const half2* wh = reinterpret_cast<const half2*>(&wv);
            const half2* xh = reinterpret_cast<const half2*>(&xv);

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                float2 a = __half22float2(wh[t]);
                float2 b = __half22float2(xh[t]);
                thread_sum += a.x * b.x + a.y * b.y;
            }
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        if (lane_id == 0) {
            y[row] = __float2half(thread_sum);
        }
    }
}


/*
 * Host-side launcher. This is what Python calls.
 *
 * Returns y (M,) fp16 on the same device as W.
 */
#ifdef TORCH_EXTENSION
static torch::Tensor gemv_fp16_launch(torch::Tensor W, torch::Tensor x, bool vectorized)
{
    TORCH_CHECK(W.is_cuda() && x.is_cuda(), "W and x must be CUDA tensors");

    TORCH_CHECK(W.dim() == 2, "W must be 2-d (matrix), got ", W.dim());
    TORCH_CHECK(x.dim() == 1, "x must be a 1-d vector, got ", x.dim());

    TORCH_CHECK(W.size(1) == x.size(0), "shape mismatch: W is (", W.size(0), ",", W.size(1), ") but x is (", x.size(0), ")");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous, kernel is row-major format");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    //row
    const int M = W.size(0);
    //col
    const int K = W.size(1);

    constexpr int THREADS = 256;
    static_assert(THREADS % 32 == 0, "warp per row mapping needs to be multiple of 32");

    constexpr int WARPS_PER_BLOCK = THREADS / 32;

    const size_t shmem = static_cast<size_t>(K) * sizeof(half);

    TORCH_CHECK(shmem <= 48 * 1024, "K=", K, " needs ", shmem, " B of shared memory, over the 48 KB default cap");

    TORCH_CHECK(!vectorized || K % 8 == 0, "vectorized path needs K % 8 == 0 (float4 = 8 halves), got K=", K);

    //output
    auto y = torch::empty({M}, W.options());
    const int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    auto W_ptr = reinterpret_cast<const half*>(W.data_ptr<at::Half>());
    auto x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    // NOT const: this is the output
    auto y_ptr = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    auto stream = at::cuda::getCurrentCUDAStream();

    if (vectorized) {
        gemv_fp16_vec_kernel<<<blocks, THREADS, shmem, stream>>>(W_ptr, x_ptr, y_ptr, M, K);
    } else {
        gemv_fp16_kernel<<<blocks, THREADS, shmem, stream>>>(W_ptr, x_ptr, y_ptr, M, K);
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

torch::Tensor gemv_fp16(torch::Tensor W, torch::Tensor x) { return gemv_fp16_launch(W, x, false); }

torch::Tensor gemv_fp16_vec(torch::Tensor W, torch::Tensor x) { return gemv_fp16_launch(W, x, true); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_fp16", &gemv_fp16, "fp16 GEMV (y = W @ x), scalar half loads");
    m.def("gemv_fp16_vec", &gemv_fp16_vec, "fp16 GEMV (y = W @ x), float4 loads");
    // step 4b: m.def("gemv_w8a16", ...)
    // step 4c: m.def("gemv_w8a16_fp8", ...)
}
#endif
