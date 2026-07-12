# ml-systems

me working through gpu and ml-systems stuff on a laptop rtx 5060 (blackwell gb206, 272 gb/s peak bandwidth). each folder is a self contained project.

the thread running through all of them is the roofline: is a given workload limited by compute or by memory bandwidth, and what do you do about it. order goes hardware -> kernels -> real model.

- **01-matmul-roofline** - measure where matmul actually lands vs the theoretical ceiling. done.
- **02-operator-fusion** - hand written cuda kernels, fused vs unfused, to see how much you save by not round tripping through memory. in progress.
- **03-quantization** - drop a model to int8/fp8 and measure the latency you gain against the quality you lose. planned.
- **04-inference-server** - batching + kv cache scheduler (a small vllm) to get decode off the memory wall. planned, this is the big one.

03 and 04 share a model runner and the prefill vs decode profiling that sits underneath both. built once, reused.

each project has its own readme and its own deps.
