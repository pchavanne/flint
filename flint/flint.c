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
flint_subtract(flint a, flint b){
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
flint_ne(flint a, flint b){
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
                        "expected one flint or float value");
        return 0;
    }
    PyObject* v[1] = {PyTuple_GET_ITEM(args,0)};

    if (PyFlint_Check(v[0])) {
        Py_INCREF(v[0]);
        return v[0];
    }
    else if (PyFloat_Check(v[0])) {
        flint f = make_flint_from_value(PyFloat_AsDouble(v[0]));
        if (PyErr_Occurred()) {
            return 0;
        }
        return PyFlint_FromFlint(f);
    }
    else {
        PyErr_Format(PyExc_TypeError,"expected float value");
        return 0;
    }
}

#define AS_FLINT(dst, object) \
    flint dst = make_flint(); \
    if (PyFlint_Check(object)){ \
        dst = ((PyFlint*)object)->f; \
    } \
    else { \
        float v_ = PyFloat_AsDouble(object); \
        if (v_==-1 && PyErr_Occurred()) { \
            if (PyErr_ExceptionMatches(PyExc_TypeError)) { \
                PyErr_Clear(); \
                Py_INCREF(Py_NotImplemented); \
                return Py_NotImplemented; \
            } \
            return 0; \
        } \
        PyObject* y_ = PyFloat_FromDouble(v_); \
        if (!y_) { \
            return 0; \
        } \
        int eq_ = PyObject_RichCompareBool(object,y_,Py_EQ); \
        Py_DECREF(y_); \
        if (eq_<0) { \
            return 0; \
        } \
        if (!eq_) { \
            Py_INCREF(Py_NotImplemented); \
            return Py_NotImplemented; \
        } \
        dst = make_flint_from_value(v_); \
    }

static PyObject*
pyflint_richcompare(PyObject* a, PyObject* b, int op) {
    AS_FLINT(x,a);
    AS_FLINT(y,b);
    int result = 0;
    #define OP(py,op) case py: result = flint_##op(x,y); break;
    switch (op) {
        OP(Py_LT,lt)
        OP(Py_LE,le)
        OP(Py_EQ,eq)
        OP(Py_NE,ne)
        OP(Py_GT,gt)
        OP(Py_GE,ge)
    };
    #undef OP
    return PyBool_FromLong(result);
}

static PyObject*
pyflint_repr(PyObject* self) {
    flint x = ((PyFlint*)self)->f;
    char* f_char =  PyOS_double_to_string(flint_double(x), 'f', x.nb_digits, 0, Py_DTST_FINITE);
    return PyUString_FromString(strcat("flint", f_char));
}

static PyObject*
pyflint_str(PyObject* self) {
    flint x = ((PyFlint*)self)->f;
    char* f_char =  PyOS_double_to_string(flint_double(x), 'f', x.nb_digits, 0, Py_DTST_FINITE);
    return PyUString_FromString(f_char);
}

static long
pyflint_hash(PyObject* self) {
    flint x = ((PyFlint*)self)->f;
    long h = x.int_value;
    /* Never return the special error value -1 */
    return h==-1?2:h;
}

#define FLINT_BINOP_2(name,exp) \
    static PyObject* \
    pyflint_##name(PyObject* a, PyObject* b) { \
        AS_FLINT(x,a); \
        AS_FLINT(y,b); \
        flint z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return PyFlint_FromFlint(z); \
    }
#define  FLINT_BINOP(name)  FLINT_BINOP_2(name,flint_##name(x,y))
FLINT_BINOP(add)
FLINT_BINOP(subtract)
FLINT_BINOP(multiply)
FLINT_BINOP(divide)

#define  FLINT_UNOP(name,type,exp,convert) \
    static PyObject* \
    pyflint_##name(PyObject* self) { \
        flint x = ((PyFlint*)self)->f; \
        type y = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return convert(y); \
    }
FLINT_UNOP(negative,flint,flint_negative(x),pyflint_FromRational)
FLINT_UNOP(absolute,flint,flint_abs(x),pyflint_FromRational)
FLINT_UNOP(int,long,flint_int(x),PyInt_FromLong)
FLINT_UNOP(float,double,flint_double(x),PyFloat_FromDouble)

static PyObject*
pyflint_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static int
pyflint_nonzero(PyObject* self) {
    flint x = ((PyFlint*)self)->f;
    return flint_nonzero(x);
}

static PyNumberMethods pyflint_as_number = {
        pyflint_add,          /* nb_add */
        pyflint_subtract,     /* nb_subtract */
        pyflint_multiply,     /* nb_multiply */
        pyflint_divide,       /* nb_divide */
        0,                    /* nb_remainder */
        0,                    /* nb_divmod */
        0,                    /* nb_power */
        pyflint_negative,     /* nb_negative */
        pyflint_positive,     /* nb_positive */
        pyflint_absolute,     /* nb_absolute */
        pyflint_nonzero,      /* nb_nonzero */
        0,                    /* nb_invert */
        0,                    /* nb_lshift */
        0,                    /* nb_rshift */
        0,                    /* nb_and */
        0,                    /* nb_xor */
        0,                    /* nb_or */
        0,                    /* nb_coerce */
        pyflint_int,          /* nb_int */
        pyflint_int,          /* nb_long */
        pyflint_float,        /* nb_float */
        0,                    /* nb_oct */
        0,                    /* nb_hex */

        0,                    /* nb_inplace_add */
        0,                    /* nb_inplace_subtract */
        0,                    /* nb_inplace_multiply */
        0,                    /* nb_inplace_divide */
        0,                    /* nb_inplace_remainder */
        0,                    /* nb_inplace_power */
        0,                    /* nb_inplace_lshift */
        0,                    /* nb_inplace_rshift */
        0,                    /* nb_inplace_and */
        0,                    /* nb_inplace_xor */
        0,                    /* nb_inplace_or */

        0,                    /* nb_floor_divide */
        pyflint_divide,       /* nb_true_divide */
        0,                    /* nb_inplace_floor_divide */
        0,                    /* nb_inplace_true_divide */
        0,                    /* nb_index */
};

static PyObject*
pyflint_int_value(PyObject* self, void* closure) {
    return PyInt_FromLong(((PyFlint*)self)->f.int_value);
}

static PyGetSetDef pyflint_getset[] = {
        {(char*)"int_value",pyflint_int_value,0,(char*)"numerator",0},
        {0} /* sentinel */
};

static PyTypeObject PyFlint_Type = {
#if defined(NPY_PY3K)
        PyVarObject_HEAD_INIT(&PyType_Type, 0)
#else
        PyObject_HEAD_INIT(&PyType_Type)
        0,                                        /* ob_size */
#endif
        "flint",                                  /* tp_name */
        sizeof(PyFlint),                          /* tp_basicsize */
        0,                                        /* tp_itemsize */
        0,                                        /* tp_dealloc */
        0,                                        /* tp_print */
        0,                                        /* tp_getattr */
        0,                                        /* tp_setattr */
#if defined(NPY_PY3K)
        0,                                         /* tp_reserved */
#else
        0,                                         /* tp_compare */
#endif
        pyflint_repr,                             /* tp_repr */
        &pyflint_as_number,                       /* tp_as_number */
        0,                                        /* tp_as_sequence */
        0,                                        /* tp_as_mapping */
        pyflint_hash,                             /* tp_hash */
        0,                                        /* tp_call */
        pyflint_str,                              /* tp_str */
        0,                                        /* tp_getattro */
        0,                                        /* tp_setattro */
        0,                                        /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
        "Fixed precision float number",           /* tp_doc */
        0,                                        /* tp_traverse */
        0,                                        /* tp_clear */
        pyflint_richcompare,                      /* tp_richcompare */
        0,                                        /* tp_weaklistoffset */
        0,                                        /* tp_iter */
        0,                                        /* tp_iternext */
        0,                                        /* tp_methods */
        0,                                        /* tp_members */
        pyflint_getset,                           /* tp_getset */
        0,                                        /* tp_base */
        0,                                        /* tp_dict */
        0,                                        /* tp_descr_get */
        0,                                        /* tp_descr_set */
        0,                                        /* tp_dictoffset */
        0,                                        /* tp_init */
        0,                                        /* tp_alloc */
        pyflint_new,                              /* tp_new */
        0,                                        /* tp_free */
        0,                                        /* tp_is_gc */
        0,                                        /* tp_bases */
        0,                                        /* tp_mro */
        0,                                        /* tp_cache */
        0,                                        /* tp_subclasses */
        0,                                        /* tp_weaklist */
        0,                                        /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
        0,                                        /* tp_version_tag */
#endif
};

#ifdef __cplusplus
}
#endif