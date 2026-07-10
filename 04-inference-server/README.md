# 04 inference server

not started yet, this is the capstone. plan below.

at batch 1, decode wastes the gpu. you load all of the weights out of hbm to produce a single token and do almost no math per byte, so you sit pinned to the bandwidth ceiling. you can't fix that with a faster kernel, the kernel is already at the ceiling. the fix is batching many requests together so each weight load gets reused across many sequences. doing that needs a scheduler and a kv cache manager, which is basically a small vllm. the server is the optimization, not scaffolding around it.

## plan

1. model runner: load an open model, kv cached generate loop. shared with 03.
2. prefill vs decode profiling: time to first token (prefill, compute bound) against time per output token (decode, memory bound), plotted on the 01 roofline. this is the "why" for everything below.
3. paged kv cache: block based allocation so many sequences share memory without fragmenting
4. scheduler: request queue with continuous batching, admit new requests and retire finished ones each step
5. batched attention over the paged cache
6. small http endpoint (fastapi) with token streaming
7. benchmark throughput vs batch size, compare against real vllm as the baseline

deliverable is a throughput vs batch size curve showing decode climbing off the memory wall, plus how far off real vllm i land.
