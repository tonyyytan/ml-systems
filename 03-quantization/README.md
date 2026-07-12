# 03 quantization

not started yet, plan below.

decode is memory bound. every token you stream the whole model out of hbm, so one way to move fewer bytes is to store the weights in fewer bits. quantize a small model to int8 and fp8, measure the latency you get back and the quality you give up.

## plan

- take a small open model (llama 3.2 1b or qwen 2.5 1.5b)
- quantize weights, and maybe activations, to int8, then fp8
- measure decode latency and per token bandwidth at each precision
- measure quality against the fp16 baseline (perplexity or a small eval)
- plot the latency vs quality tradeoff across precisions
- stretch: implement turboquant

builds on the shared model runner and the prefill/decode profiling (see 04 and the repo readme). this is the numerics side of attacking the memory wall. 04 is the systems side.
