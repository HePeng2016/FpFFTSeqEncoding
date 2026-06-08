from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fft_cuda',
    ext_modules=[
        CUDAExtension(
            name='fft_cuda',
            sources=[
                'fft_bindings.cpp',
                'fft_cuda.cu'
            ],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++17'],
                'nvcc': ['-O2', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    extra_compile_args={
    'cxx': ['-O2', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1'],
    'nvcc': ['-O2', '--use_fast_math', '-D_GLIBCXX_USE_CXX11_ABI=1']
     }
)

