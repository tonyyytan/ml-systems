# ml-systems

me working through gpu and ml-systems stuff on a laptop rtx 5060 (blackwell gb206, 384 gb/s peak bandwidth). each folder is a self contained project.

the thread running through all of them is the roofline: is a given workload limited by compute or by memory bandwidth, and what do you do about it. order goes hardware -> kernels -> real model.

| project | what | status |
|---|---|---|
| [01-matmul-roofline](01-matmul-roofline) | measure where matmul actually lands vs the theoretical ceiling | done, plot + numbers in the readme |
| [02-operator-fusion](02-operator-fusion) | hand written cuda kernels, fused vs unfused, how much you save by not round tripping through memory | done, benchmarks in the readme |
| [03-inference-engine](03-inference-engine) | run models bigger than 8gb by deciding what lives in vram, what lives in ram, and when to move it | in progress, this is the big one |
| [04-gpu-clusters] (04-gpu-clusters) |	scale out to rented multi-gpu setups, dealing with network bottlenecks and distributed training | in progress, testing out clusters |

03 is where it all lands. decode is stuck against the bandwidth ceiling: you stream the whole model out of memory to make a single token and barely do any math with it. spill the model to system ram and it gets worse, because pcie is another 10x slower than vram, so there are now two rooflines and the interesting one is the slow one.

the deliverable is the engine: a server that unifies vram and system ram into one space so a model too big for the card runs at all, still in progress. the **two tier roofline** is the tool that predicts the payoff before you run anything: quantization stops being "2x fewer bytes" and becomes a cliff at the vram boundary, and batching stops being about serving many users and becomes the only way to manufacture enough compute to hide a pcie transfer behind. the roofline says which optimization pays; the engine is what delivers it.

04 is where we leave the laptop. when the dataset or model is too big for single-node offloading, you rent a cluster. but now the roofline has a third tier: the network. compute scales linearly, but you're trading a pcie bottleneck for an nvlink, infiniband, or ethernet bottleneck across nodes.

the deliverable is the cluster setup: terraform/docker scripts to spin up, configure, and tear down distributed training (fsdp/deepspeed) and multi-gpu inference on rented iron. the distributed roofline is the tool here: if your all-reduce communication takes longer than your forward pass, adding more gpus actually slows you down. the roofline dictates when to split the model across layers (pipeline parallelism) vs splitting the math (tensor parallelism); the cluster is where you test it.

each project has its own readme and its own deps.
