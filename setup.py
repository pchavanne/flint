from distutils.core import setup, Extension
import numpy as np

flint_ext = Extension('flint',
                       sources=['flint.c'],
                       include_dirs=[np.get_include()],
                       extra_compile_args=['-std=c99'])

setup(name='flint',
      version='0.1',
      description='float with int arithmetics type extensions',
      packages=['flint'],
      ext_modules=flint_ext)

