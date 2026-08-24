/*
 * Standalone bandwidth harness for gemv.cu. CSV on stdout, progress on stderr.
 *
 * GEMV at batch 1 is 1 flop/byte, so the only figure of merit is achieved
 * bandwidth: GB/s = M*K*sizeof(weight) / seconds.
 *
 * Lock the clocks before trusting a sweep -- this is a laptop card and it
 * throttles:  sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc <mhz>
 *
 * Build: make      Run: make run
 */

// Makefile compiles only this file, so the kernel arrives by include.
// TORCH_EXTENSION is undefined here, so gemv.cu's torch launcher is excluded.
#include "gemv.cu"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t _s = (call);                                              \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, \
                         static_cast<int>(_s));                                  \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static constexpr int WARMUP_ITERS = 20;
static constexpr int TIMED_ITERS  = 100;

static constexpr size_t MIN_FLUSH_BYTES = 128ull << 20;
static constexpr size_t STREAM_BYTES    = 512ull << 20;

struct Shape {
    const char* name;
    int M;
    int K;
};

// llama-3-8b decode shapes: hidden 4096, ffn 14336, GQA 8 kv heads.
static const Shape SHAPES[] = {
    {"q_proj",    4096,  4096},
    {"kv_proj",   1024,  4096},
    {"o_proj",    4096,  4096},
    {"gate_up",  14336,  4096},
    {"down",      4096, 14336},
};


/*
 * The practical bandwidth ceiling: how fast can this card read memory when the
 * kernel does nothing else? Every gemv number below is reported as a fraction
 * of whatever this achieves, so it has to be the best possible reader.
 */
__global__ void stream_read_kernel(const float4* __restrict__ src, float* __restrict__ sink, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float acc{};

    for(; i < n; i+= stride) {

        float4 vals = src[i];
        acc += vals.x + vals.y + vals.z + vals.w;
    }

    // never true at runtime, but the compiler can't prove it, so the loads stay alive
    if (acc == 1.2345678e30f) {
        sink[0] = acc;
    }
}


struct Timing {
    double median_ms;
    double min_ms;
};

template <typename LaunchFn>
static Timing time_kernel(LaunchFn&& launch, void* flush_buf, size_t flush_bytes)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> samples;
    samples.reserve(TIMED_ITERS);

    for (int i = 0; i < TIMED_ITERS; ++i) {
        // Evict L2 so we time DRAM, not cache. Ordered before the kernel but
        // outside the event window, so it costs wall clock, not measured time.
        CUDA_CHECK(cudaMemsetAsync(flush_buf, i & 0xff, flush_bytes));

        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(static_cast<double>(ms));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return Timing{samples[samples.size() / 2], samples.front()};
}

// Won't match bit-for-bit: different reduction order, fp add isn't associative.
// The tolerance is sized to catch stride and tail bugs, not rounding.
static double max_rel_err(const std::vector<half>& got, const std::vector<half>& ref)
{
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double g = static_cast<double>(__half2float(got[i]));
        double r = static_cast<double>(__half2float(ref[i]));
        double denom = std::fabs(r) > 1e-3 ? std::fabs(r) : 1e-3;
        worst = std::max(worst, std::fabs(g - r) / denom);
    }
    return worst;
}

/*
 * Host-side quantizer. Mirrors engine/quantize.py EXACTLY -- fp32 amax per row,
 * s = max(amax / 127, 1e-8), round-half-to-even, clamp to +/-127. If the two
 * rounding rules drift apart, kernel-vs-reference error is dominated by the
 * mismatch and you go hunting a kernel bug that does not exist.
 *
 * Also returns W_hat = s * q in fp16. The KERNEL never materializes this; it
 * exists so cuBLAS can produce a reference the int8 kernel is checked against,
 * which keeps kernel bugs separate from quantization error.
 */
static constexpr int QMAX = 127;
static constexpr float SCALE_EPS = 1e-8f;

static void quantize_w8a16(const std::vector<half>& W, int M, int K, std::vector<int8_t>& q, std::vector<half>& s, std::vector<half>& W_hat)
{
    q.resize(static_cast<size_t>(M) * K);
    s.resize(M);
    W_hat.resize(static_cast<size_t>(M) * K);

    for (int m = 0; m < M; ++m) {
        const size_t base = static_cast<size_t>(m) * K;

        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            amax = std::max(amax, std::fabs(__half2float(W[base + k])));
        }

        const float scale = std::max(amax / QMAX, SCALE_EPS);
        s[m] = __float2half(scale);

        // read the scale back through fp16: that is what the kernel sees
        const float scale_h = __half2float(s[m]);

        for (int k = 0; k < K; ++k) {
            float level = std::nearbyint(__half2float(W[base + k]) / scale);
            level = std::min(static_cast<float>(QMAX), std::max(-static_cast<float>(QMAX), level));

            q[base + k] = static_cast<int8_t>(level);
            W_hat[base + k] = __float2half(scale_h * level);
        }
    }
}


/*
 * Norm-relative error, ||got - ref|| / ||ref||. This is quantize.py's out_rel.
 *
 * max_rel_err above is elementwise, and at batch 1 y is a sum of random-sign
 * products, so some rows land near zero and their elementwise relative error
 * explodes regardless of how good the kernel is. That is fine for catching
 * stride and tail bugs, where every row is wrong, and useless for pricing
 * quantization, where the vector as a whole is what the next layer sees.
 */
static double rel_l2_err(const std::vector<half>& got, const std::vector<half>& ref)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = static_cast<double>(__half2float(got[i]));
        const double r = static_cast<double>(__half2float(ref[i]));
        num += (g - r) * (g - r);
        den += r * r;
    }
    return std::sqrt(num) / std::sqrt(den > 0.0 ? den : 1.0);
}


int main()
{
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Cross-check on the hardcoded peak. If these disagree, trust neither yet.
    const double derived_peak_gb_s =
        2.0 * static_cast<double>(prop.memoryClockRate) * 1e3 *
        (static_cast<double>(prop.memoryBusWidth) / 8.0) / 1e9;

    std::fprintf(stderr, "device      : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::fprintf(stderr, "L2 cache    : %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    std::fprintf(stderr, "peak (spec) : %.1f GB/s\n", PEAK_BW_GB_S);
    std::fprintf(stderr, "peak (props): %.1f GB/s\n", derived_peak_gb_s);
    std::fprintf(stderr, "warmup/iters: %d / %d\n\n", WARMUP_ITERS, TIMED_ITERS);

    size_t flush_bytes = std::max(MIN_FLUSH_BYTES, static_cast<size_t>(prop.l2CacheSize) * 4);
    void* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    float4* d_stream = nullptr;
    float*  d_sink   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stream, STREAM_BYTES));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_stream, 1, STREAM_BYTES));

    const size_t stream_n = STREAM_BYTES / sizeof(float4);

    Timing t_stream = time_kernel(
        [&] { stream_read_kernel<<<1024, 256>>>(d_stream, d_sink, stream_n); },
        d_flush, flush_bytes);

    const double stream_gbs_med = STREAM_BYTES / (t_stream.median_ms * 1e-3) / 1e9;
    const double practical_peak_gb_s = STREAM_BYTES / (t_stream.min_ms * 1e-3) / 1e9;

    CUDA_CHECK(cudaFree(d_stream));

    std::fprintf(stderr, "practical ceiling: %.1f GB/s (%.1f%% of spec)\n\n",
                 practical_peak_gb_s, 100.0 * practical_peak_gb_s / PEAK_BW_GB_S);

    std::printf("shape,variant,M,K,bytes,flops,ms_median,ms_min,"
                "gbs_median,gbs_min,pct_spec,pct_practical,max_rel_err,rel_l2_err\n");
    std::printf("-,stream_read,0,0,%zu,0,%.6f,%.6f,%.2f,%.2f,%.2f,100.00,0,0\n",
                STREAM_BYTES, t_stream.median_ms, t_stream.min_ms,
                stream_gbs_med, practical_peak_gb_s,
                100.0 * practical_peak_gb_s / PEAK_BW_GB_S);
    std::fflush(stdout);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (const Shape& s : SHAPES) {
        const int M = s.M;
        const int K = s.K;

        const size_t w_elems  = static_cast<size_t>(M) * K;
        const size_t w_bytes  = w_elems * sizeof(half);
        const double bytes_read = static_cast<double>(w_bytes);
        const double flops      = 2.0 * static_cast<double>(M) * K;

        // mirrors gemv_fp16_kernel: one warp per output row, x tiled in dynamic shared
        constexpr int THREADS = 256;
        constexpr int WARPS_PER_BLOCK = THREADS / 32;

        const size_t shmem = static_cast<size_t>(K) * sizeof(half);

        if (K % 16 != 0) {
            std::fprintf(stderr, "skip %s: K=%d not a multiple of 16 (vec path needs float4 rows, int8 path int4)\n", s.name, K);
            continue;
        }

        if (shmem > 48 * 1024) {
            std::fprintf(stderr, "skip %s: K=%d needs %zu B shared, over the 48 KB cap\n", s.name, K, shmem);
            continue;
        }

        const int blocks_gemv = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        // Small magnitudes keep the fp32 accumulator inside fp16 range after K adds.
        std::vector<half> h_W(w_elems);
        std::vector<half> h_x(K);
        for (size_t i = 0; i < w_elems; ++i) {
            h_W[i] = __float2half((static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f);
        }
        for (int i = 0; i < K; ++i) {
            h_x[i] = __float2half((static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f);
        }

        half *d_W = nullptr, *d_x = nullptr, *d_y = nullptr, *d_y_ref = nullptr;
        CUDA_CHECK(cudaMalloc(&d_W, w_bytes));
        CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y_ref, M * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice));

        // quantize on the host, exactly as quantize.py does it
        std::vector<int8_t> h_q;
        std::vector<half> h_s, h_W_hat;
        quantize_w8a16(h_W, M, K, h_q, h_s, h_W_hat);

        int8_t* d_q = nullptr;
        half* d_s = nullptr;
        half* d_W_hat = nullptr;
        CUDA_CHECK(cudaMalloc(&d_q, w_elems * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_s, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_W_hat, w_bytes));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), w_elems * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), M * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W_hat, h_W_hat.data(), w_bytes, cudaMemcpyHostToDevice));

        // what the int8 kernel actually reads: q, plus one scale per row
        const double q_bytes_read = static_cast<double>(w_elems) + static_cast<double>(M) * sizeof(half);

        // cuBLAS is column-major, so our row-major (M,K) reads as column-major
        // (K,M): OP_T with lda=K recovers (M,K). n=1 makes the gemm a matvec.
        const float alpha = 1.0f, beta = 0.0f;
        auto launch_cublas = [&] {
            CUBLAS_CHECK(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                M, 1, K,
                &alpha,
                d_W, CUDA_R_16F, K,
                d_x, CUDA_R_16F, K,
                &beta,
                d_y_ref, CUDA_R_16F, M,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };

        // cuBLAS has no plain fp16 gemv; the only one is the batched form, so
        // batchCount=1. HSH = half in, fp32 compute, half out, and alpha/beta
        // are float* for that variant. A is stored (m,n) = (K,M), so OP_T.
        auto launch_cublas_gemv = [&] {
            CUBLAS_CHECK(cublasHSHgemvStridedBatched(
                handle, CUBLAS_OP_T,
                K, M,
                &alpha,
                d_W, K, 0,
                d_x, 1, 0,
                &beta,
                d_y, 1, 0,
                1));
        };

        auto launch_gemv = [&] {
            gemv_fp16_kernel<<<blocks_gemv, THREADS, shmem>>>(d_W, d_x, d_y, M, K);
        };

        auto launch_gemv_vec = [&] {
            gemv_fp16_vec_kernel<<<blocks_gemv, THREADS, shmem>>>(d_W, d_x, d_y, M, K);
        };

        auto launch_w8a16 = [&] {
            gemv_w8a16_kernel<<<blocks_gemv, THREADS, shmem>>>(d_q, d_s, d_x, d_y, M, K);
        };

        launch_cublas();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        std::vector<half> h_y(M), h_y_ref(M);
        CUDA_CHECK(cudaMemcpy(h_y_ref.data(), d_y_ref, M * sizeof(half), cudaMemcpyDeviceToHost));

        // one correctness check per variant, each against the same cuBLAS output
        struct Err { double max_rel; double rel_l2; };

        auto check = [&](const char* variant, auto&& launch) {
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost));
            Err e{max_rel_err(h_y, h_y_ref), rel_l2_err(h_y, h_y_ref)};
            if (e.rel_l2 > 5e-3) {
                std::fprintf(stderr, "WARNING %s/%s: rel l2 err %.4f vs cuBLAS\n", s.name, variant, e.rel_l2);
            }
            return e;
        };

        const Err err     = check("fp16_scalar", launch_gemv);
        const Err err_vec = check("fp16_vec", launch_gemv_vec);
        const Err err_cbv = check("cublas_gemv", launch_cublas_gemv);

        // The int8 kernel cannot be checked against the fp16 answer -- it is
        // computing a different (quantized) matrix on purpose. Run cuBLAS on
        // W_hat to get the answer the kernel SHOULD produce, so the check sees
        // reduction-order rounding only. The accuracy price of quantizing is a
        // separate number, measured just below against the fp16 reference.
        std::vector<half> h_y_qref(M);
        {
            half* d_W_saved = d_W;
            d_W = d_W_hat;
            launch_cublas();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_y_qref.data(), d_y_ref, M * sizeof(half), cudaMemcpyDeviceToHost));
            d_W = d_W_saved;
        }

        // the accuracy price of quantizing, measured the way quantize.py does
        const double quant_out_err = rel_l2_err(h_y_qref, h_y_ref);

        launch_w8a16();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost));
        const Err err_q{max_rel_err(h_y, h_y_qref), rel_l2_err(h_y, h_y_qref)};
        if (err_q.rel_l2 > 5e-3) {
            std::fprintf(stderr, "WARNING %s/w8a16: rel l2 err %.4f vs dequantized cuBLAS\n", s.name, err_q.rel_l2);
        }

        // W_hat was scaffolding for that one reference. Free it before timing so
        // it is not sitting in memory competing for the L2 flush.
        CUDA_CHECK(cudaFree(d_W_hat));

        Timing t_gemv       = time_kernel(launch_gemv, d_flush, flush_bytes);
        Timing t_gemv_vec   = time_kernel(launch_gemv_vec, d_flush, flush_bytes);
        Timing t_w8a16      = time_kernel(launch_w8a16, d_flush, flush_bytes);
        Timing t_cublas     = time_kernel(launch_cublas, d_flush, flush_bytes);
        Timing t_cublas_gv  = time_kernel(launch_cublas_gemv, d_flush, flush_bytes);

        // bytes differ per variant now: the int8 path reads half the weight bytes
        // plus the (M,) scales, and GB/s has to be computed on what was actually read.
        auto emit = [&](const char* variant, const Timing& t, const Err& e, double vbytes) {
            const double gbs_med = vbytes / (t.median_ms * 1e-3) / 1e9;
            const double gbs_min = vbytes / (t.min_ms * 1e-3) / 1e9;
            std::printf("%s,%s,%d,%d,%.0f,%.0f,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.3e,%.3e\n",
                        s.name, variant, M, K, vbytes, flops,
                        t.median_ms, t.min_ms, gbs_med, gbs_min,
                        100.0 * gbs_min / PEAK_BW_GB_S,
                        100.0 * gbs_min / practical_peak_gb_s,
                        e.max_rel, e.rel_l2);
        };

        emit("fp16_scalar", t_gemv, err, bytes_read);
        emit("fp16_vec", t_gemv_vec, err_vec, bytes_read);
        emit("w8a16", t_w8a16, err_q, q_bytes_read);
        emit("cublas_gemm", t_cublas, Err{0.0, 0.0}, bytes_read);
        emit("cublas_gemv", t_cublas_gv, err_cbv, bytes_read);
        std::fflush(stdout);

        const double ours_gbs   = bytes_read / (t_gemv.min_ms * 1e-3) / 1e9;
        const double vec_gbs    = bytes_read / (t_gemv_vec.min_ms * 1e-3) / 1e9;
        const double cublas_gbs = bytes_read / (t_cublas.min_ms * 1e-3) / 1e9;
        const double q_gbs      = q_bytes_read / (t_w8a16.min_ms * 1e-3) / 1e9;
        std::fprintf(stderr, "%-9s w8a16 %6.1f GB/s (%4.1f%%)  %.3f ms vs %.3f vec = %.2fx   "
                             "kernel err %.1e   quant err %.1e\n",
                     s.name, q_gbs, 100.0 * q_gbs / practical_peak_gb_s,
                     t_w8a16.min_ms, t_gemv_vec.min_ms, t_gemv_vec.min_ms / t_w8a16.min_ms,
                     err_q.rel_l2, quant_out_err);
        std::fprintf(stderr, "%-9s M=%5d K=%5d  scalar %6.1f GB/s (%4.1f%%)   "
                             "vec %6.1f GB/s (%4.1f%%)   cublas %6.1f GB/s   err %.1e / %.1e\n",
                     s.name, M, K,
                     ours_gbs, 100.0 * ours_gbs / practical_peak_gb_s,
                     vec_gbs, 100.0 * vec_gbs / practical_peak_gb_s,
                     cublas_gbs, err.rel_l2, err_vec.rel_l2);

        CUDA_CHECK(cudaFree(d_W));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_y_ref));
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_s));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_flush));
    return 0;
}
