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
                "cxx": ["-O3", "-std=c++17", "-DTORCH_EXTENSION"],
                "nvcc": ["-O3", "-arch=sm_120", "-std=c++17", "-diag-suppress=177", "-DTORCH_EXTENSION"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
