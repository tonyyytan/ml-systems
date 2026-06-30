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
            # TODO: add nvcc flags here if needed, e.g.:
            # extra_compile_args={"nvcc": ["-O3", "-arch=sm_120"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
