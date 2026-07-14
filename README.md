# ml-systems

me working through gpu and ml-systems stuff on a laptop rtx 5060 (blackwell gb206, 272 gb/s peak bandwidth). each folder is a self contained project.

the thread running through all of them is the roofline: is a given workload limited by compute or by memory bandwidth, and what do you do about it. order goes hardware -> kernels -> real model.

- **01-matmul-roofline** - measure where matmul actually lands vs the theoretical ceiling. done.
- **02-operator-fusion** - hand written cuda kernels, fused vs unfused, to see how much you save by not round tripping through memory. done.
- **03-inference-engine** - run models bigger than 8gb by deciding what lives in vram, what lives in ram, and when to move it. in progress, this is the big one.

03 is where it all lands. decode is stuck against the bandwidth ceiling: you stream the whole model out of memory to make a single token and barely do any math with it. spill the model to system ram and it gets worse, because pcie is another 10x slower than vram, so there are now two rooflines and the interesting one is the slow one.

the deliverable isn't the engine, it's the **two tier roofline**: a model of this machine that predicts which optimization pays and by how much, before you run anything. quantization stops being "2x fewer bytes" and becomes a cliff at the vram boundary. batching stops being about serving many users and becomes the only way to manufacture enough compute to hide a pcie transfer behind. the engine exists to test that model.

each project has its own readme and its own deps.
