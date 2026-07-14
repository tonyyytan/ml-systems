# 03 inference engine

in progress. this is the capstone.

## what it is

an inference engine for an 8gb laptop card that runs models **bigger than its own vram**, by deciding what lives in vram, what lives in system ram, and when to move it.

the engine is the vehicle. the actual deliverable is a **two tier roofline**: a model of this machine that predicts, before you run anything, which optimization is going to pay and by how much.

## the wall, and why offload changes its shape

decode is memory bound. to make one token you stream every weight in the model out of memory, do a trivial amount of arithmetic with each one, and throw it away. roughly 1-2 flops per byte, so you sit pinned to the 272 gb/s ceiling with the compute units idle. a faster kernel does nothing. the kernel is already at the ceiling.

now make the model too big to fit. the spilled layers go to system ram, which is not on the fast road, it's across pcie at somewhere around 16-32 gb/s. **an offloaded layer is about ten times slower to stream than a resident one.**

so there is no longer one roofline, there are two:

```
decode time  ~=  bytes in vram / 272 gb/s  +  bytes in ram / ~25 gb/s
```

the second term swamps the first almost immediately, and that changes what optimization means.

### the cliff

**quantization's payoff is not linear in bytes, it's a step function at the vram boundary.**

while the model already fits, int8 halves the traffic and buys you ~2x. fine. but if quantizing is what stops the model spilling *at all*, you don't get 2x, you get 5-10x, because you deleted the pcie term from the equation. the win came from crossing a boundary, not from moving fewer bytes.

8gb puts that cliff exactly where it can be studied. llama 3 8b: fp16 is ~16gb and spills badly, int8 is ~8gb and sits on the knife edge, int4 is ~4gb and fits with room for kv cache. the cliff is sweepable on hardware i already own.

### why batching comes back

offloading naively serialises: copy layer i, compute layer i, copy layer i+1, compute layer i+1. the fix is to prefetch layer i+1 on a separate cuda stream while the gpu computes layer i. if compute time >= transfer time the copy is free, fully hidden.

but **at batch 1 there is almost no compute to hide behind.** that's the whole point of decode being memory bound. so prefetch buys nearly nothing.

what manufactures compute to hide the transfer behind? **batching.** not to serve many users, there's only one user here. to raise arithmetic intensity until the pcie copy disappears under the math.

which means the whole project collapses into one argument:

> model doesn't fit -> offload -> now pcie bound -> raise arithmetic intensity until compute hides the transfer -> that is exactly what quantization and batching do

three escapes from the memory wall, all in service of one goal, on one machine. that's the paper.

## the contribution

not a new algorithm. offloading is well trodden (flexgen, headinfer, specoffload) and every serious engine already has paging and continuous batching (vllm, sglang, exllamav3, mlc). **not claiming to have invented any of it.**

what doesn't exist is the model. everyone who runs local models hits this cliff and reasons about it by folklore ("just fit it in vram"). nobody writes down the equation and checks it. so:

> given a bandwidth budget, a vram budget, a batch size and a sequence length: where does the cliff sit, which knob should you reach for, and does the roofline predict the answer before you run it?

the mechanism that makes it work is a **tiered, quantized block allocator**. location and precision are both properties of a block, not global settings:

- hot kv blocks: vram, fp16
- aged blocks: demote precision, then evict to ram
- weights: per-layer placement, prefetched on a copy stream

that composes paging, quantization and offload into one mechanism instead of three bolted together features. that part is mine.

## baselines

three of them, and they answer different questions:

- **`cudaMallocManaged`** - let the driver page-migrate on demand. the *no policy* control. beating it shows a policy beats no policy.
- **llama.cpp `-ngl`** - a fixed, hand-chosen layer split. the state of the practice. beating it is the real claim.
- **vllm** - the ceiling, with speculative decoding **off** so it's like for like. it has mtp/eagle3/dflash and i don't; leaving them on measures the absence of a feature i never intended to build, not the quality of my engine.

everything exposes the same `generate(prompts, max_tokens)` so `bench.py` sweeps them uniformly.

## scope: the claim vs the engine

the full engine (runner, quantized gemv, paged cache, paged attention kernel, scheduler, placement layer) is months of solo work, and paged attention alone is a serious kernel. **most of it is not required to prove the claim.** so the line is drawn here, on purpose, for whichever future version of me is short on time:

**load bearing. no result without these:**

- the two tier roofline itself. it's arithmetic, a couple hundred lines.
- the measurement harness: pcie bandwidth, tokens/sec, bytes moved.
- placement + prefetch. the custom core, the part that is actually mine.
- the tiered block allocator, location and precision per block.

**borrowable. building them teaches me things, but the claim survives without them:**

- the runner. hf already has one and i only need a floor.
- paged attention. start with a contiguous cache and pytorch sdpa. paging only has to exist once memory pressure is real, and flashinfer's kernel is right there.
- the scheduler. static batching is enough to get the throughput curve, continuous batching is a refinement.
- quantized gemv. `torchao` can stand in while the model is being validated. writing the kernel is the fun part, not the load bearing part.

**so the engine is the last thing i build, not the first.** each piece gets added when a measurement demands it, and every addition is justified by a number i already collected.

### the failure mode to avoid

not that any one piece is too hard. it's spending two months on a paged attention kernel, getting blocked or bored, and never writing down the roofline result that was the actually novel bit and would have taken a week.

**steps 0-2 need almost no engine code and are a complete result on their own.** build the cheap result first, then build the engine to make it a strong one.

## build order

**0. measure the machine.** half a day, before any engine code.
- pcie bandwidth, host-to-device, pinned vs pageable (`cudaMemcpy` microbenchmark). **this number is the second slope of the entire project.**
- **pinned memory on wsl2.** async prefetch needs page-locked host memory (`cudaHostAlloc`). wsl2 gpu passthrough has been quirky here. if it can't hit full speed the prefetch mechanism is dead and i need native linux. find out now, not in week six.
- does vllm build on sm_120. blackwell consumer support has been finicky and vllm is the ceiling for everything.

**1. two tier roofline** - `roofline2.py`
extend the 01 plot with the pcie slope. predict decode throughput as a function of model size, precision, and fraction of layers offloaded. **this is the artifact. everything below exists to test it.**

**2. validate against llama.cpp** - cheap and high signal
sweep `-ngl` from 0 to all layers on a model that doesn't fit. measure tokens/sec at each point. overlay the prediction from step 1. if the curve lands, the model is real and the rest of the project has a foundation. if it doesn't, stop and find out why.

**3. model runner** - `runner.py`
load a model, kv cached greedy generate. the floor. two modes: batch 1, and static batching (pad to longest, wait for the slowest, contiguous max-length kv per sequence). the static mode matters: without it the final chart can only prove "batching helps", which nobody doubts, instead of "my batching is good", which is the actual claim.

**4. quantized gemv** - `gemv.cu`, `quantize.py`
fp16 baseline first so there's a number to beat, then w8a16: int8 weights, fp16 activations, per-channel scales, dequant fused in-register so a weight only ever crosses the bus as one byte. then fp8 e4m3 (sm_120 has native conversion, so int8 vs fp8 is a measurement here, not a guess).

**5. offload + prefetch** - `placement.py`
per-layer placement across vram/ram, double buffered, `cudaMemcpyAsync` on a dedicated copy stream. measure transfer/compute overlap directly. **expect it to disappoint at batch 1.** that's not a bug, it's the finding that motivates step 7.

**6. tiered paged kv cache** - `paged_cache.py`, `attention.cu`
block allocator where each block carries a location and a precision. attention reads a non-contiguous, mixed-precision cache. at 8gb the cache genuinely runs out, so eviction is a real decision instead of a design doc.

**7. scheduler** - `scheduler.py`
continuous batching. admit and retire requests every step. here batching stops being about serving many users and starts being about generating enough compute to hide pcie behind.

**8. bench + plot** - `bench.py`, `plot_engine.py`
sweep batch size x precision x offload fraction across all systems. record where each one ooms. overlay every measurement on the step 1 prediction.

**9. stretch: speculative decoding**
the escape that works at batch 1. use the int4 model as its own draft, verify with fp16. prior work exists (ml-specqd, quantspec), build on it, don't re-derive it.

## layout

kernels in cuda, harness in python. same split as 02: `.cu` files build into one torch extension via `setup.py`.

```
roofline2.py     the two tier model. the actual deliverable.
gemv.cu          fp16 / int8 / fp8 fused dequant matvec
attention.cu     attention over the tiered paged cache
setup.py         builds the extension
quantize.py      per-channel scales + weight packing
placement.py     vram/ram layer placement + prefetch streams
paged_cache.py   block allocator, location + precision per block
scheduler.py     continuous batching
runner.py        the floor: batch 1 + static batching
engine.py        mine: ties placement, cache, scheduler and kernels together
bench.py         sweep over all systems
plot_engine.py   the charts
```

## notes

- per-channel scales, not per-tensor. per-tensor is easier and loses more than it needs to.
- quality side: perplexity delta vs fp16 on wikitext, so every latency win carries a price tag.
- unified memory machines (apple silicon, amd strix halo) have no pcie hop at all, moving a layer is a page table update, not a copy. **the cliff does not exist there.** worth one paragraph of contrast, it explains why macs punch above their weight.
