from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext = CppExtension(
    'mla_kernel',
    sources=['main.cpp', 'mla_kernel.cu'],
    include_dirs=['./'],
)

setup(
    name='mla_kernel',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension}
)