from distutils.core import setup, Extension
import numpy as np

flint_ext = Extension('flint',
                       sources=['flint/flint.c'],
                       include_dirs=[np.get_include()])

setup(name='flint',
      version='0.1',
      description='Python and Numpy float type with integer arithmetic',
      packages=['flint'],
      author='Philippe Chavanne',
      author_email = 'philippe.chavanne@gmail.com',
      url = 'https://github.com/pchavanne/flint',
      ext_modules=[flint_ext])

