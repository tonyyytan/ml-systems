"""
Build the CUDA kernels as a PyTorch C++ extension so benchmark.py can import them.

Usage:
    python setup.py build_ext --inplace

This produces activations_cuda.so (or similar) in this directory.
After building, in Python:
    import activations_cuda
    y = activations_cuda.relu_fwd(x)

How this works:
  - CUDAExtension tells setuptools to compile kernels.cu with nvcc
  - TORCH_EXTENSION_NAME (set by the build) becomes the module name
  - The PYBIND11_MODULE block at the bottom of kernels.cu registers the ops
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="activations_cuda",
    ext_modules=[
        CUDAExtension(
            name="activations_cuda",
            sources=["kernels.cu"],
            extra_compile_args={
                "cxx": ["-03", "-std=c++17"],
                "nvcc": ["-03", "-arch=sm_120", "-std=c++17", "-diag-suppress=177"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
