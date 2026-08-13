"""
Builds the CUDA extension for project 03, importable as `gemv_cuda`.

Lives at the TOP level (not next to the .cu files) on purpose: it needs to see
every kernel in engine/kernels/ so attention.cu can be added to `sources` in
step 6 without a second build file. Same pattern as 02, different scope.

Usage, from THIS directory:
    python3 setup.py build_ext --inplace

That drops a gemv_cuda*.so here, which `import gemv_cuda` then picks up.

Version-matching rule (see requirements.txt): torch's CUDA major must equal the
local nvcc major. nvcc 12.9 -> a cu12x torch wheel. -arch=sm_120 is Blackwell
(RTX 5060 laptop, GB206); it is not portable to other cards.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemv_cuda",
    ext_modules=[
        CUDAExtension(
            name="gemv_cuda",
            sources=["engine/kernels/gemv.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DTORCH_EXTENSION"],
                "nvcc": ["-O3", "-arch=sm_120", "-std=c++17", "-diag-suppress=177", "-DTORCH_EXTENSION"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
