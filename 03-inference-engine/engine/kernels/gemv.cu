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

// torch/extension.h must come first when building as an extension
#ifdef TORCH_EXTENSION
#include <torch/extension.h>
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
    // int col = 1 (uninitialized)

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
 * Host-side launcher. This is what Python calls.
 *
 * Returns y (M,) fp16 on the same device as W.
 */
#ifdef TORCH_EXTENSION
torch::Tensor gemv_fp16(torch::Tensor W, torch::Tensor x)
{
    
    // TODO 5. Validate inputs with TORCH_CHECK before touching raw pointers.
    //         A wrong assumption here shows up as garbage numbers or a crash
    //         deep in the kernel, which is miserable to debug. Check:
    //           - both are CUDA tensors
    //           - both are kHalf
    //           - W is 2-D, x is 1-D
    //           - W.size(1) == x.size(0)
    //           - W.is_contiguous() -- the kernel assumes row-major packing

    // TODO 6. Allocate the output:
    //         torch::empty({M}, W.options())   inherits dtype + device from W.

    // TODO 7. Pick the launch config and call the kernel.
    //         blocks = M, threads = 256.
    //         Cast data pointers with W.data_ptr<at::Half>() then
    //         reinterpret_cast<const half*>(...) -- at::Half and half are the
    //         same 2 bytes but distinct C++ types.

    // TODO 8. Check for launch errors: cudaGetLastError(). A bad launch config
    //         fails silently otherwise and you get zeros back.

    return torch::Tensor();  // replace
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_fp16", &gemv_fp16, "fp16 GEMV (y = W @ x), batch-1 decode shape");
    // step 4b: m.def("gemv_w8a16", ...)
    // step 4c: m.def("gemv_w8a16_fp8", ...)
}
#endif
