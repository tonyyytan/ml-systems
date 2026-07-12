# 02 operator fusion

point of this one is to see how much you save by fusing elementwise and normalization ops instead of launching them separately and paying for an extra read + write to global memory each time. these ops are all memory bound, so the win comes from moving fewer bytes, not from doing less math.

## kernels

- relu
- bias add + gelu
- residual add + layernorm

each has an unfused version (two kernels, extra round trip through memory) and a fused version (one kernel). all of them use float4 vectorized loads and grid striding. layernorm does a warp reduction for mean and variance.

two ways to run them:

- standalone: `kernels.cu` compiles to a cli that prints csv, `plot_fusion.py` draws the chart
- as a pytorch extension: `setup.py` builds it, `benchmark.py` compares my kernels against pytorch eager and torch.compile

## status

kernels are written and the relu path runs end to end. still to do:

- `run_bias_gelu` and `run_add_layernorm` host drivers (the benchmark loops are stubbed)
- the pytorch binding functions (`relu_fwd`, `gelu_fwd` etc are stubs)
- `benchmark.py` timing loop and the silu / gelu benches
- no silu kernel exists yet, the python side already references one

## run

standalone:

```
make && ./kernels > results.csv && python3 plot_fusion.py results.csv
```

extension:

```
python setup.py build_ext --inplace && python benchmark.py
```

needs torch for the extension path.
