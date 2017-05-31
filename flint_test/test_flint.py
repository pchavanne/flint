# -*- coding: UTF-8 -*-
from flint_type import flint
import numpy as np

v1 = 123.45
v2 = 678.90

f1 = flint(v1)
f2 = flint(v2)

another_f1 = flint(v1)
assert another_f1 == f1
assert another_f1 is not f1

# Copy constructor
f_from_f1 = flint(f1)
assert f_from_f1 == f1
assert f_from_f1 is not f1

assert float(f1) == v1
assert int(f1) == v1 * f1.multiplier
assert str(f1) == "{0:.4f}".format(v1)

v3 = v1 + v2
f3 = flint(v3)
assert f3 == f1 + f2
assert f3 == f1 + v2
assert f3 == v1 + f2

v4 = v2 - v1
f4 = flint(v4)  # 555.4499999999999
f4_true = flint(round(v4, 2))   # 555.45
assert f4_true == f2 - f1
assert f4_true == v2 - f1
assert f4_true == f2 - v1

v5 = v1 * v2
f5 = flint(v5)
assert f5 == f1 * f2
assert f5 == f1 * v2
assert f5 == v1 * f2

v6 = v2 / v1
f6 = flint(v6)
assert f6 == f2 / f1
assert f6 == v2 / f1
assert f6 == f2 / v1

arr = [v1, v2]
npf = np.asarray(arr, dtype=flint)
npf_from_flint = np.asarray([f1, f2], dtype=flint)
assert (npf == npf_from_flint).all()
assert npf.shape == (2,)
npf.mean()
npf.cumsum()

