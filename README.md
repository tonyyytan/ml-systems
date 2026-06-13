# ml-systems

**Project A:** Roofline plot for square fp32 and fp16 matmul on a 5060 laptop gpu, with measured TFLOPS overlaid on the theoretical ceiling.

**Project B:** Three cuda kernels benchmarked against Pytorch's defaults (aim to understand operator fusion)

**Project C:** Measure and characterize the prefill phase vs the decode phase of an open source transformer model and optimize inference on it -> Bonus: Quantize the model to int8 or fp8 and measure latency/quality tradeoff/implement TurboQuant.
