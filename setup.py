from distutils.core import setup, Extension
import numpy as np

flint_ext = Extension('flint_type.flint',
                      sources=['flint_type/flint.c'],
                      include_dirs=[np.get_include()])

setup(name='flint_type',
      version='0.1.1',
      description='Python and Numpy float type with integer arithmetic',
      packages=['flint_type'],
      author='Philippe Chavanne',
      author_email='philippe.chavanne@gmail.com',
      url='https://github.com/pchavanne/flint_type',
      install_requires=['numpy'],
      ext_modules=[flint_ext])

