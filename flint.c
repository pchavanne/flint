/* Float with integer arithmetic*/

#define NPY_NO_DEPRECATED_API NPY_API_VERSION


#include <Python.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Relevant arithmetic exceptions */

/* Uncomment the following line to work around a bug in numpy */
/* #define ACQUIRE_GIL */

static void
set_overflow(void) {
#ifdef ACQUIRE_GIL
    /* Need to grab the GIL to dodge a bug in numpy */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_OverflowError,
                "overflow in rational arithmetic");
    }
#ifdef ACQUIRE_GIL
    PyGILState_Release(state);
#endif
}

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

static NPY_INLINE int64_t
safe_abs64(int64_t x) {
    if (x>=0) {
        return x;
    }
    int64_t nx = -x;
    if (nx<0) {
        set_overflow();
    }
    return nx;
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
    if (b.int_value == 0){
        set_zero_divide():
    }
    else{
        div.int_value = (int64_t)(a.int_value * DEFAULT_MULTIPLIER / b.int_value);
    }
    return div;
}

static flint
flint_abs(flint f){
    flint abs = make_flint();
    abs.int_value = safe_abs64(f.int_value);
    return abs;
}

static int
flint_sign(flint f){
    return f.int_value<0?-1:f.int_value==0?0:1;
}

static int64_t
flint_int(flint f){
    return f.int_value;
}

static double
flint_double(flint f){
    return (double)f.int_value / f.multiplier;
}

static int
flint_nonzero(flint f){
    return f.int_value!=0;
}

/* Expose flint to Python as a numpy scalar */

typedef struct {
    PyObject_HEAD;
    flint f;
} PyFlint;

static PyTypeObject PyFlint_Type;

static NPY_INLINE int
PyFlint_Check(PyObject* object) {
    return PyObject_IsInstance(object,(PyObject*)&PyFlint_Type);
}

static PyObject*
PyFlint_FromFlint(flint f) {
    PyFlint* p = (PyFlint*)PyFlint_Type.tp_alloc(&PyFlint_Type,0);
    if (p) {
        p->f = f;
    }
    return (PyObject*)p;
}


#ifdef __cplusplus
}
#endif