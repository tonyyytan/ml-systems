"""
Build kernels.cu as a PyTorch C++ extension (importable as activations_cuda).

Usage:
    python setup.py build_ext --inplace
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
