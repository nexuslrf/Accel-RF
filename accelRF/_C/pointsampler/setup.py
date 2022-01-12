from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import glob

# specify src
_ext_src_root = "./"
_ext_sources = \
    glob.glob("{}/*.cpp".format(_ext_src_root)) + \
    glob.glob("{}/*.cu".format(_ext_src_root))
_ext_headers = glob.glob("../include/*.h")

# Standalone package
setup(
    name='accelRF_C_pointsampler',
    ext_modules=[
        CUDAExtension(
            name='_ext',
            sources=_ext_sources,
            extra_compile_args={
                'cxx': ['-O2', '-ffast-math', '-I../include'], 
                'nvcc': ['-O2', '-I../include']})
    ],
    cmdclass={ 
        'build_ext' : BuildExtension 
    }
)