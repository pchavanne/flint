import numpy as np

import flint

if np.__dict__.get('rational') is not None:
    raise RuntimeError('The NumPy package already has a rational type')

np.flint = flint
np.typeDict['flint'] = np.dtype(flint)