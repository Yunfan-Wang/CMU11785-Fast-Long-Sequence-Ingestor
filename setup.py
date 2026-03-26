from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Define the C++ and CUDA sources as I have previously planned 
sources = [
    'csrc/attention_api.cpp',
    'csrc/attention_kernel.cu'
]

setup(
    name='custom_attention_hpc',
    ext_modules=[
        CUDAExtension(
            name='custom_attention_hpc',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)