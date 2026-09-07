# 04 gpu clusters

status: planned. this is where we leave the laptop.

when the model or dataset gets too big for single-node offloading — or when you actually need to train instead of just infer — you rent a cluster. but scaling compute isn't free; it just moves the bottleneck further out. 

## what it is

a set of tools and infrastructure-as-code to spin up, configure, benchmark, and tear down distributed multi-gpu environments on rented iron (runpod, lambda labs, etc.), without burning money debugging drivers manually.

the deliverable is twofold: **the cluster setup** (terraform/docker to get 8x gpus talking to each other predictably) and the **distributed roofline** (a tool that predicts whether splitting the model will actually speed you up, before you pay per hour to find out).

## the wall: the third roofline tier

in `03`, we dealt with two tiers: vram (fast) and pcie (slow). renting a cluster introduces a third tier: **the network.** 

compute scales linearly with node count, but communication overhead grows. if you split a model across GPUs, they have to sync. you are trading a 14 gb/s laptop pcie bottleneck for an interconnect bottleneck that varies wildly depending on what you paid for:

| tier | bandwidth | where you find it | latency |
|---|---|---|---|
| **nvlink** | 600 - 900 gb/s | inside an 8x h100 node | microseconds |
| **infiniband** | 50 - 100 gb/s | high-end multi-node clusters | microseconds |
| **ethernet** | 1 - 10 gb/s | cheap rented multi-node | milliseconds |

**the trap:** if your `all-reduce` communication takes longer than your forward pass, adding more gpus actually slows you down. 

this forces the layout decision. **tensor parallelism (TP)** splits individual matrix multiplications, requiring massive bandwidth because GPUs sync *every layer*. **pipeline parallelism (PP)** puts different layers on different GPUs, requiring syncs only at the boundaries, but introduces "bubble" idle time. 

the distributed roofline predicts which split to use based on the rented interconnect. if you rent a cheap ethernet cluster and try to run TP, you will spend 95% of your time waiting on the network. the model tells you that before you boot the instance.

## build order

**0. measure the network** 
run `nccl-tests` (specifically `all_reduce_perf` and `sendrecv_perf`) to map the actual topology. rented nodes often lie about their bandwidth or isolate GPUs into different NUMA nodes. this script pins down the exact interconnect speed.

**1. the distributed roofline**
a simple model extending `01` and `03`. given model dimensions, batch size, and the `nccl` bandwidth numbers, output the expected throughput for TP vs PP. output the optimal split.

**2. infrastructure as code**
terraform scripts to provision nodes, plus a docker setup that installs the exact right CUDA, PyTorch, and NCCL versions. **goal: 5 minutes from `terraform apply` to a running distributed training job.** you pay by the minute, so environment setup has to be automated.

**3. distributed runner**
a simple FSDP (Fully Sharded Data Parallel) or DeepSpeed training loop. strictly a test payload to prove the cluster is utilizing all GPUs at the throughput the roofline predicted.

## layout

```text
infrastructure/        spin up, spin down
  terraform/           node provisioning
  docker/              nccl/cuda environment

model/                 deliverable: the network predictor
  nccl_harness.py      runs nccl-tests, parses topology
  roofline_dist.py     predicts TP vs PP crossover based on bandwidth

runner/                the payload
  train_fsdp.py        proof of life distributed training loop
```