import numpy as np
from flint_type import flint

__all__ = ['flint']

np.flint = flint
np.typeDict['flint'] = np.dtype(flint)