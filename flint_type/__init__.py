import numpy as np
from flint_type.flint import flint

__all__ = ['flint']

if np.__dict__.get('flint') is not None:
    raise RuntimeError('The NumPy package already has a flint type')

np.flint = flint
np.typeDict['flint'] = np.dtype(flint)
