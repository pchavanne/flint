#ifndef __FLINT_H__
#define __FLINT_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	long val;
	int multiplier;
	int digits;
} flint;

int flint_isnonzero(flint f);
int flint_isnan(flint f);
int flint_isinf(flint f);
int flint_isfinite(flint f);
double flint_absolute(flint f);
flint flint_add(flint f1, flint f2);
flint flint_subtract(flint f1, flint f2);
flint flint_multiply(flint f1, flint f2);
flint flint_divide(flint f1, flint f2);
flint flint_multiply_scalar(flint f, double s);
flint flint_divide_scalar(flint f, double s);
flint flint_negative(flint f);
int flint_equal(flint f1, flint f2);
int flint_not_equal(flint f1, flint f2);
int flint_less(flint f1, flint f2);
int flint_less_equal(flint f1, flint f2);


#ifdef __cplusplus
}
#endif

#endif