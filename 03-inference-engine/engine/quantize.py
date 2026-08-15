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
    """Per-channel symmetric int8. Returns (q, s).

    in   W : (M, K) fp16 or fp32, any device
    out  q : (M, K) int8, contiguous, row-major   <- what the kernel reads
         s : (M,)   fp16                          <- one scale per output row

    Steps:
      1. amax = |W| max over dim=1, computed in FP32. This runs once, and the
         division that follows is precision-sensitive in a way the max is not.
      2. s = amax / QMAX, clamped up to SCALE_EPS.
      3. q = round(W_fp32 / s[:, None]) clamped to [-QMAX, QMAX], cast to int8.
      4. return q.contiguous() and s.half().

    torch.round is round-half-to-even, which is what the C++ side gets from
    rintf. Do not swap it for floor(x + 0.5).
    """
    raise NotImplementedError  # TODO


def dequantize_w8a16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Reconstruct W_hat = s * q. Returns (M, K) fp16.

    Only the reference path uses this -- the KERNEL never materializes W_hat,
    that is the entire point. This exists so quant_error has something to
    measure and so bench_gemv has a correctness target.

    Multiply in fp32, then cast down. s[:, None] broadcasts over k.
    """
    raise NotImplementedError  # TODO


def quant_error(W: torch.Tensor, W_hat: torch.Tensor, x: torch.Tensor | None = None) -> dict:
    """Three numbers, measuring three different things.

    max_abs_rel   max |W_hat - W| / max|W|. Worst single weight. Will look bad
                  (~1/255 = 0.4%) and is the least informative of the three.
    rms_rel       ||W_hat - W|| / ||W||. The representative elementwise number.
    out_rel       ||W_hat@x - W@x|| / ||W@x||, plus cosine similarity, for a
                  random x. THIS is the one that predicts model quality, and it
                  should come out far better than max_abs_rel because K
                  independent rounding errors partially cancel (~sqrt(K)
                  suppression). The gap between out_rel and max_abs_rel is a
                  result worth putting in the README.

    Compute all of it in fp32. If x is None, draw a standard normal (K,).
    """
    raise NotImplementedError  # TODO


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
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    main()
