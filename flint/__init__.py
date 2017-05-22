import numpy as np

from flint import flint

np.flint = flint
np.typeDict['flint'] = np.dtype(flint)
