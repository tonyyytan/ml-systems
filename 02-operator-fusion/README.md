# 02 operator fusion

point of this one is to see how much you save by fusing elementwise and normalization ops instead of launching them separately and paying for an extra read + write to global memory each time. these ops are all memory bound, so the win comes from moving fewer bytes, not from doing less math.

## kernels

- bias add + gelu
- residual add + layernorm

each has an unfused version (two kernels, extra round trip through memory) and a fused version (one kernel). all of them use float4 vectorized loads and grid striding. layernorm does a warp reduction for mean and variance.

relu is also here but it's a single op, there is nothing to fuse. it's the bandwidth reference: the number a trivial kernel gets is the ceiling everything else is measured against.

two ways to run them:

- standalone: `kernels.cu` compiles to a cli that prints csv, `plot_fusion.py` draws the chart
- as a pytorch extension: `setup.py` builds it, `benchmark.py` compares my kernels against pytorch eager and torch.compile

## status

done. both paths run end to end, charts are in `fusion_benchmark.png` (standalone) and `bench_benchmark.png` (vs torch).

loose ends, none of them blocking:

- everything is fp32. `benchmark.py` still has a `torch.float16` todo and the kernels are float4-only. fp16 halves the bytes moved, which is the most direct test of the whole premise, so this is the one worth doing.
- the fused layernorm reads x and residual twice, once for the reduction and once for the normalize pass, but the bandwidth math only charges it one read. the second read probably hits l2 so it isn't a full hbm round trip, but the kernel isn't genuinely single pass. staging the row in registers or shared would make it one.
- `gelu_fwd` is exported to python but nothing calls it. `bias_gelu_fwd` is the one that gets used.

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
