#include "flint.h"
#include "math.h"

#define _EPS 1e-6

int
flint_isnonzero(flint f)
{
    return f.value != 0;
}

int
flint_isnan(flint f)
{
    return isnan(f.value);
}

int
flint_isinf(flint f)
{
    return isinf(f.value);
}

int
flint_isfinite(flint f)
{
    return isfinite(f.value);
}

double
flint_absolute(flint f)
{
   return abs(f.value);
}

flint
flint_add(flint f1, flint f2)
{
   return (flint) {f1.value + f2.value, f1.multiplier, f1.digits};
}

flint
flint_subtract(flint f1, flint f2)
{
   return (flint) {f1.value - f2.value, f1.multiplier, f1.digits};
}

flint
flint_multiply(flint f1, flint f2)
{
   return (flint) {double(f1.value * f2.value / f1.multiplier), f1.multiplier, f1.digits};
}

flint
flint_divide(flint f1, flint f2)
{
   return (flint) {double(f1.value * f1.multiplier / f2.value), f1.multiplier, f1.digits};
}

flint
flint_multiply_scalar(flint f, float s)
{
   return (flint) {double(f1.value * s / f1.multiplier), f1.multiplier, f1.digits};
}

flint
flint_divide_scalar(flint f, float s)
{
   return (flint) {double(f1.value * f1.multiplier / s), f1.multiplier, f1.digits};
}

flint
flint_negative(flint f)
{
   return (flint) {-f1.value, f1.multiplier, f1.digits};
}

int
flint_equal(flint f1, flint f2)
{
    return !flint_isnan(f1) && !flint_isnan(f2)) && f1.value == f2.value && f1.multiplier == f2.multiplier;
}

int
flint_not_equal(flint f1, flint f2)
{
    return !flint_equal(f1, f2);
}

int
flint_less(flint f1, flint f2)
{
    return !flint_isnan(f1) && !flint_isnan(f2)) && f1.value < f2.value;
}

int
flint_less_equal(flint f1, flint f2)
{
   return !flint_isnan(f1) && !flint_isnan(f2)) && f1.value <= f2.value;
}