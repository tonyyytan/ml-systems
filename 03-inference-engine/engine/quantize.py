"""
Project 03, Step 4a: weight quantization. THE LAYOUT CONTRACT.

This file exists before the kernel on purpose. It decides what
gemv_w8a16_kernel is going to read out of memory -- dtype, shape, and the exact
arithmetic that turns int8 back into a number. Write the kernel first and you
are guessing at all three.

W8A16: weights int8, activations fp16. At batch 1 the weights ARE the traffic
(M*K bytes vs 2*K for x), so halving the weight bytes is the whole speedup.
Quantizing activations too (W8A8) buys int8 tensor cores, which is a compute
win, and step 4 is not compute bound. Not worth it here.

    per-channel symmetric int8, absmax round-to-nearest

Taken apart, because each word is a decision with a kernel consequence:

  symmetric      W ~= s * q,  no zero point. The asymmetric form W ~= s*(q-z)
                 is actually cheap too -- it expands to
                     s[m] * (sum_k q[m][k]*x[k]  -  z[m] * sum_k x[k])
                 and sum_k x[k] is ONE scalar for the whole block, computable
                 while staging x into shared memory. But weights are roughly
                 zero-centred, so the extra range buys ~nothing. Symmetric.

  per-channel    one scale per OUTPUT ROW, i.e. amax over dim=1 (the k axis).
                 This is the choice that makes the kernel free:
                     y[m] = sum_k (s[m] * q[m][k]) * x[k]
                          = s[m] * sum_k q[m][k] * x[k]
                 s[m] does not vary down the k loop, so it factors out of the
                 dot product entirely. ONE multiply per row, after the warp
                 reduction -- not one per element. That is what "fused dequant"
                 means here: no dequantized weight is ever written to memory.

                 The alternative, per-group along K (llama.cpp's Q8_0 uses
                 groups of 32), is more accurate but s changes down the loop, so
                 you need a partial accumulator per group. Necessary for int4.
                 Not necessary for int8.

  absmax RTN     s = max|W| / 127, round to nearest. No calibration data, no
                 search. One outlier weight stretches the scale and wastes
                 levels, but per-channel granularity confines that damage to a
                 single row. The ladder above this: percentile clipping, MSE
                 clip search, AWQ (activation-aware rescaling), GPTQ (Hessian
                 error compensation). All need calibration data. Out of scope.

CONSISTENCY TRAP: bench_gemv.cu quantizes on the host in C++ and must use the
IDENTICAL rule -- fp32 amax, divide, round-half-to-even, clamp to +/-127. If the
two disagree, kernel-vs-reference error is dominated by a rounding mismatch and
you will hunt a kernel bug that does not exist. This docstring is the spec; the
C++ mirrors it.
"""

import torch


# Symmetric int8 uses 127 levels per side, NOT 128. Using -128 breaks the
# symmetry (s*-128 has no positive twin) to buy one extra level.
QMAX = 127

# Floor for the scale so an all-zero row does not divide by zero.
SCALE_EPS = 1e-8


def quantize_w8a16(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #fp32 for accurate calc
    W_fp32 = W.to(torch.float32)

    #abs max
    amax = W_fp32.abs().max(dim=1).values

    #scale factor
    s = torch.clamp(amax / QMAX, min = SCALE_EPS)

    #new scaled weights
    q = torch.clamp(torch.round(W_fp32 / s[:, None]), min = -QMAX, max = QMAX)
    q = q.to(torch.int8)

    return q.contiguous(), s.half()

def dequantize_w8a16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

    """Reconstruct W_hat = s * q. Returns (M, K) fp16.

    Only the reference path uses this -- the KERNEL never materializes W_hat,
    that is the entire point. This exists so quant_error has something to
    measure and so bench_gemv has a correctness target.

    Multiply in fp32, then cast down. s[:, None] broadcasts over k.
    """

    q_fp32 = q.to(torch.float32)
    s_fp32 = s.to(torch.float32)

    W_hat = s_fp32[:, None] * q_fp32
    W_hat = W_hat.to(torch.float16)

    return W_hat


def quant_error(W: torch.Tensor, W_hat: torch.Tensor, x: torch.Tensor | None = None) -> dict:
    """Three numbers, measuring three different things.

    max_abs_rel   max |W_hat - W| / max|W|. Worst single weight. Will look bad
                  (~1/255 = 0.4%) and is the least informative of the three.
    rms_rel       ||W_hat - W|| / ||W||. The representative elementwise number.
    out_rel       ||W_hat@x - W@x|| / ||W@x||, plus cosine similarity, for a
                  random x. THIS is the one that predicts model quality.
                  It comes out equal to rms_rel, NOT better: the K rounding
                  errors do add in quadrature and grow as sqrt(K), but ||W@x||
                  grows as sqrt(K) too, so the ratio is unchanged. Averaging
                  buys absolute accuracy, not relative. cos_sim is the number
                  worth quoting (~0.99997) since direction is what the next
                  layer sees, and it is far more forgiving than the magnitudes.

    Compute all of it in fp32. If x is None, draw a standard normal (K,).
    """
    #fp32 so the error is not itself rounded
    W_fp32 = W.to(torch.float32)
    W_hat_fp32 = W_hat.to(torch.float32)

    err = W_hat_fp32 - W_fp32

    #worst single weight, normalised by the largest weight in the tensor
    max_abs_rel = (err.abs().max() / W_fp32.abs().max()).item()

    #the representative elementwise number
    rms_rel = (err.norm() / W_fp32.norm()).item()

    if x is None:
        x = torch.randn(W.size(1), generator = torch.Generator().manual_seed(0))

    x_fp32 = x.to(torch.float32)

    #what the kernel actually produces, where the rounding errors cancel
    y = W_fp32 @ x_fp32
    y_hat = W_hat_fp32 @ x_fp32

    out_rel = ((y_hat - y).norm() / y.norm()).item()
    cos_sim = torch.nn.functional.cosine_similarity(y_hat, y, dim = 0).item()

    return {
        "max_abs_rel": max_abs_rel,
        "rms_rel": rms_rel,
        "out_rel": out_rel,
        "cos_sim": cos_sim,
    }


# llama-3-8b per-layer projections, d_model=4096, d_ffn=14336, 8 kv heads.
# Same five shapes bench_gemv.cu sweeps, so the accuracy table lines up with
# the bandwidth table row for row.
SHAPES = [
    ("q_proj", 4096, 4096),
    ("kv_proj", 1024, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up", 14336, 4096),
    ("down", 4096, 14336),
]


def main():
    """Print the accuracy price tag, one row per shape.

    Random normal weights are a stand-in for real ones and will slightly
    FLATTER the result -- real weight rows have heavier tails, so absmax
    stretches further. Note that in the README rather than pretending it does
    not matter; re-run against real llama weights once runner.py loads them.

    Seed the RNG so the table is reproducible.
    """
    torch.manual_seed(0)

    print(f"{'shape':<10} {'M':>6} {'K':>6} {'max_abs_rel':>12} {'rms_rel':>10} {'out_rel':>10} {'cos_sim':>10}")

    for name, M, K in SHAPES:
        W = torch.randn(M, K, dtype = torch.float16)

        q, s = quantize_w8a16(W)
        W_hat = dequantize_w8a16(q, s)

        e = quant_error(W, W_hat)

        print(f"{name:<10} {M:>6} {K:>6} {e['max_abs_rel']:>12.5f} {e['rms_rel']:>10.5f} {e['out_rel']:>10.5f} {e['cos_sim']:>10.6f}")


if __name__ == "__main__":
    main()
