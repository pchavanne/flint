import numpy as np

import flint
from flint.info import __doc__

__all__ = ['denominator', 'gcd', 'lcm', 'numerator', 'rational']

if np.__dict__.get('rational') is not None:
    raise RuntimeError('The NumPy package already has a rational type')

np.flint = flint
np.typeDict['flint'] = np.dtype(flint)