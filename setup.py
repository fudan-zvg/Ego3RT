from setuptools import find_packages, setup

import os
import shutil
import sys
import torch
import warnings
from os import path as osp
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)



if __name__ == '__main__':

    setup(
        name='projects',
        version='1.0.0',
        description='plug-in sub packages to official codebase',
        author='SirJamie',
        packages=['projects'],
        ext_modules=[
            make_cuda_ext(
                name='msda_ext',
                module='projects.mmdet3d_plugin.ops.msda',
                sources=[
                    'src/vision.cpp',
                    'src/cpu/ms_deform_attn_cpu.cpp',
                ],
                sources_cuda=[
                    'src/cuda/ms_deform_attn_cuda.cu'
                ],
                extra_args=[
                    '-DCUDA_HAS_FP16=1'
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
    )

