/* Float with integer arithmetic*/

#define NPY_NO_DEPRECATED_API NPY_API_VERSION


#include <Python.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static void
set_zero_divide(void) {
#ifdef ACQUIRE_GIL
    /* Need to grab the GIL to dodge a bug in numpy */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                        "zero divide in rational arithmetic");
    }
#ifdef ACQUIRE_GIL
    PyGILState_Release(state);
#endif
}

static const int64_t DEFAULT_MULTIPLIER = 1000000;
static const int61_t DEFAULT_NB_DIGITS = 4;

typedef struct {
    int64_t int_value;
    int64_t multiplier;
    int16_t nb_digits;
} flint;

static flint
make_flint(void){
    flint f = {
            .int_value = 0,
            .multiplier = DEFAULT_MULTIPLIER,
            .nb_digits = DEFAULT_NB_DIGITS
    };
    return f;
}

static flint
make_flint(float float_value) {
    flint f = make_flint();
    f.int_value = (int64_t)(float_value * DEFAULT_MULTIPLIER);
    return f;
}

static flint
flint_negative(flint f){
    flint neg = make_flint();
    neg.int_value = -f.int_value;
    return neg;
}

static flint
flint_add(flint a, flint b){
    flint add = make_flint();
    add.int_value = a.int_value + b.int_value;
    return add;
}

static flint
flint_substract(flint a, flint b){
    flint sub = make_flint();
    sub.int_value = a.int_value - b.int_value;
    return sub;
}

static flint
flint_multiply(flint a, flint b){
    flint mult = make_flint();
    mult.int_value = (int64_t)(a.int_value * b.int_value / DEFAULT_MULTIPLIER);
    return mult;
}

static flint
flint_divide(flint a, flint b){
    flint div = make_flint();
    div.int_value = (int64_t)(a.int_value * DEFAULT_MULTIPLIER / b.int_value);
    return div;
}


#ifdef __cplusplus
}
#endif