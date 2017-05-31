 flint
========

Python and Numpy float type with integer arithmetic

**flint = (int) (float * default_multiplier)**

a flint is a float multiplied by a multiplier (default is 1,000,000) and casted to an integer
it avoids the float precision problem:

    >>> 3 * 3.2
    9.600000000000001

Using the flint type solve the problem:

    >>> from flint_type import flint
    >>> float(flint(3)*flint(3.2))
    9.6

flints are displayed with a default precision of 4 digits

    >>> f = flint(3.2)
    >>> f
    3.2000
    >>> str(f)
    '3.2000'
    
the internal integer representation is accessible directly or by casting to int

    >>> f.int_value
    3200000L
    >>> int(f)
    3200000

flint can be casted to float

    >>> float(f)
    3.2

flint allows scalar arithmetic

    >>> f*3
    9.6000
    
flint has a numpy dtype

    >>> import numpy as np
    >>> npf = np.asarray([1, 2, 3.5, 4.5], dtype=np.flint)
    >>> npf
    array([1.0000, 2.0000, 3.5000, 4.5000], dtype=flint)

with usual numpy usage

    >>> npf.shape
    (4,)
    >>> 2*npf
    array([2.0000, 4.0000, 7.0000, 9.0000], dtype=flint)
    >>> npf**2
    array([1.0000, 4.0000, 12.2500, 20.2500], dtype=flint)
    >>> npf.mean()
    2.7500
    >>> npf.cumsum()
    array([1.0000, 3.0000, 6.5000, 11.0000], dtype=flint)

    
### installation

    pip install flint_type

or the latest

    pip install git+https://github.com/pchavanne/flint_type
