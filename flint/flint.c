/* Float with integer arithmetic*/

#define NPY_NO_DEPRECATED_API NPY_API_VERSION


#include <Python.h>

#include <stdint.h>
#include <math.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include "numpy/npy_3kcompat.h"

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
static const int64_t DEFAULT_NB_DIGITS = 4;

typedef struct {
    int64_t int_value;
    int64_t multiplier;
    int16_t nb_digits;
} flint;

static NPY_INLINE flint
make_flint(void){
    flint f = {
            .int_value = 0,
            .multiplier = DEFAULT_MULTIPLIER,
            .nb_digits = DEFAULT_NB_DIGITS
    };
    return f;
}

static flint
make_flint_from_value(float float_value) {
    flint f = make_flint();
    f.int_value = (int64_t)(float_value * DEFAULT_MULTIPLIER);
    return f;
}

static NPY_INLINE flint
flint_negative(flint f){
    flint neg = make_flint();
    neg.int_value = -f.int_value;
    return neg;
}

static NPY_INLINE flint
flint_add(flint a, flint b){
    flint add = make_flint();
    add.int_value = a.int_value + b.int_value;
    return add;
}

static NPY_INLINE flint
flint_substract(flint a, flint b){
    flint sub = make_flint();
    sub.int_value = a.int_value - b.int_value;
    return sub;
}

static NPY_INLINE flint
flint_multiply(flint a, flint b){
    flint mult = make_flint();
    mult.int_value = (int64_t)(a.int_value * b.int_value / DEFAULT_MULTIPLIER);
    return mult;
}

static NPY_INLINE flint
flint_divide(flint a, flint b){
    flint div = make_flint();
    if (b.int_value == 0){
        set_zero_divide();
    }
    else{
        div.int_value = (int64_t)(a.int_value * DEFAULT_MULTIPLIER / b.int_value);
    }
    return div;
}

static NPY_INLINE flint
flint_abs(flint f){
    flint abs = make_flint();
    abs.int_value = safe_abs64(f.int_value);
    return abs;
}

static NPY_INLINE int
flint_sign(flint f){
    return f.int_value<0?-1:f.int_value==0?0:1;
}

static NPY_INLINE int
flint_nonzero(flint f){
    return f.int_value!=0;
}

static NPY_INLINE int
flint_eq(flint a, flint b){
    return a.int_value==b.int_value;
}

static NPY_INLINE int
flint_neq(flint a, flint b){
    return !flint_eq(a, b);
}

static NPY_INLINE int
flint_lt(flint a, flint b){
    return a.int_value<b.int_value;
}

static NPY_INLINE int
flint_gt(flint a, flint b){
    return flint_lt(b, a);
}

static NPY_INLINE int
flint_le(flint a, flint b){
    return !flint_lt(b, a);
}

static NPY_INLINE int
flint_ge(flint a, flint b){
    return !flint_lt(a, b);
}


static NPY_INLINE int64_t
flint_int(flint f){
    return f.int_value;
}

static NPY_INLINE double
flint_double(flint f){
    return (double)f.int_value / f.multiplier;
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

static PyObject*
pyflint_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                        "constructor takes no keyword arguments");
        return 0;
    }
    Py_ssize_t size = PyTuple_GET_SIZE(args);
    if (size>1) {
        PyErr_SetString(PyExc_TypeError,
                        "expected one float value");
        return 0;
    }
    PyObject* v[1] = {PyTuple_GET_ITEM(args,0)};
    if (size==1) {
        if (PyFlinr_Check(v[0])) {
            Py_INCREF(v[0]);
            return v[0];
        }
        else if (PyString_Check(x[0])) {
            const char* s = PyString_AS_STRING(x[0]);
            rational x;
            if (scan_rational(&s,&x)) {
                const char* p;
                for (p = s; *p; p++) {
                    if (!isspace(*p)) {
                        goto bad;
                    }
                }
                return PyRational_FromRational(x);
            }
            bad:
            PyErr_Format(PyExc_ValueError,
                         "invalid rational literal '%s'",s);
            return 0;
        }
    }
    long n[2]={0,1};
    int i;
    for (i=0;i<size;i++) {
        n[i] = PyInt_AsLong(x[i]);
        if (n[i]==-1 && PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                             "expected integer %s, got %s",
                             (i ? "denominator" : "numerator"),
                             x[i]->ob_type->tp_name);
            }
            return 0;
        }
        /* Check that we had an exact integer */
        PyObject* y = PyInt_FromLong(n[i]);
        if (!y) {
            return 0;
        }
        int eq = PyObject_RichCompareBool(x[i],y,Py_EQ);
        Py_DECREF(y);
        if (eq<0) {
            return 0;
        }
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                         "expected integer %s, got %s",
                         (i ? "denominator" : "numerator"),
                         x[i]->ob_type->tp_name);
            return 0;
        }
    }
    flint f = make_flint_rom_value(v[0]);
    if (PyErr_Occurred()) {
        return 0;
    }
    return PyFlint_FromFlint(f);
}

#ifdef __cplusplus
}
#endif