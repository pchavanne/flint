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
                "overflow in flint arithmetic");
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
                        "zero divide in flint arithmetic");
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
        set_overflaw();
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

static flint
make_flint_int(int64_t n) {
    flint f = make_flint();
    f.int_value = n * DEFAULT_MULTIPLIER;
    return f;
}

static NPY_INLINE int64_t
flint_int(flint f){
return f.int_value;
}

static NPY_INLINE double
flint_double(flint f){
    return (double)f.int_value / f.multiplier;
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

static NPY_INLINE int64_t
flint_floor(flint x){
    if (x.int_value>=0){
        return flint_int(x);
    }
    else {
        return flint_int(x) -1;
    }
}

static NPY_INLINE int64_t
flint_ceil(flint x) {
    return -flint_floor(flint_negative(x));
}

static NPY_INLINE flint
flint_remainder(flint a, flint b){
    flint remainder = make_flint();
    remainder.int_value = a.int_value % b.int_value;
    return remainder;
}

static NPY_INLINE flint
flint_abs(flint f){
    flint abs = make_flint();
    abs.int_value = safe_abs64(f.int_value);
    return abs;
}

static NPY_INLINE int64_t
flint_rint(flint f){
    return rint(flint_double(f));
}

static NPY_INLINE int
flint_sign(flint f){
    return f.int_value<0?-1:f.int_value==0?0:1;
}

static NPY_INLINE flint
flint_inverse(flint f){
    flint inv = make_flint();
    if (f.int_value==0){
        set_zero_divide();
    }
    else{
        inv.int_value = (int64_t)(f.multiplier * f.multiplier / f.int_value);
    }
    return inv;
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
        double v_ = PyFloat_AsDouble(object); \
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
FLINT_BINOP(remainder)

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
FLINT_UNOP(negative,flint,flint_negative(x),PyFlint_FromFlint)
FLINT_UNOP(absolute,flint,flint_abs(x),PyFlint_FromFlint)
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
        pyflint_remainder,    /* nb_remainder */
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

/* Numpy support */

static PyObject*
npyflint_getitem(void* data, void* arr) {
    flint f;
    memcpy(&f,data,sizeof(flint));
    return PyFlint_FromFlint(f);
}

static int
npyflint_setitem(PyObject* item, void* data, void* arr) {
    flint f;
    if (PyFlint_Check(item)) {
        f = ((PyFlint*)item)->f;
    }
    else {
        double v = PyFloat_AsDouble(item);
        if (v==-1 && PyErr_Occurred()) {
            return -1;
        }
        PyObject* y = PyFloat_FromDouble(v);
        if (!y) {
            return -1;
        }
        int eq = PyObject_RichCompareBool(item,y,Py_EQ);
        Py_DECREF(y);
        if (eq<0) {
            return -1;
        }
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected flint, got %s", item->ob_type->tp_name);
            return -1;
        }
        f = make_flint_from_value(v);
    }
    memcpy(data,&f,sizeof(flint));
    return 0;
}

static NPY_INLINE void
byteswap(int64_t* x) {
    char* p = (char*)x;
    size_t i;
    for (i = 0; i < sizeof(*x)/2; i++) {
        int j = sizeof(*x)-1-i;
        char t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
}

static void
npyflint_copyswapn(void* dst_, npy_intp dstride, void* src_,
                      npy_intp sstride, npy_intp n, int swap, void* arr) {
    char *dst = (char*)dst_, *src = (char*)src_;
    if (!src) {
        return;
    }
    npy_intp i;
    if (swap) {
        for (i = 0; i < n; i++) {
            flint* f = (flint*)(dst+dstride*i);
            memcpy(f,src+sstride*i,sizeof(flint));
            byteswap(&f->int_value);
        }
    }
    else if (dstride == sizeof(flint) && sstride == sizeof(flint)) {
        memcpy(dst, src, n*sizeof(flint));
    }
    else {
        for (i = 0; i < n; i++) {
            memcpy(dst + dstride*i, src + sstride*i, sizeof(flint));
        }
    }
}

static void
npyflint_copyswap(void* dst, void* src, int swap, void* arr) {
    if (!src) {
        return;
    }
    flint* f = (flint*)dst;
    memcpy(f,src,sizeof(flint));
    if (swap) {
        byteswap(&f->int_value);
    }
}

static int
npyflint_compare(const void* d0, const void* d1, void* arr) {
    flint x = *(flint*)d0,
            y = *(flint*)d1;
    return flint_lt(x,y)?-1:flint_eq(x,y)?0:1;
}


#define FIND_EXTREME(name,op) \
    static int \
    npyflint_##name(void* data_, npy_intp n, npy_intp* max_ind, void* arr) { \
        if (!n) { \
            return 0; \
        } \
        const flint* data = (flint*)data_; \
        npy_intp best_i = 0; \
        flint best_r = data[0]; \
        npy_intp i; \
        for (i = 1; i < n; i++) { \
            if (flint_##op(data[i],best_r)) { \
                best_i = i; \
                best_r = data[i]; \
            } \
        } \
        *max_ind = best_i; \
        return 0; \
    }
FIND_EXTREME(argmin,lt)
FIND_EXTREME(argmax,gt)

static void
npyflint_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
                void* op, npy_intp n, void* arr) {
    flint f = make_flint();
    const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
    npy_intp i;
    for (i = 0; i < n; i++) {
        f = flint_add(f,flint_multiply(*(flint*)ip0,*(flint*)ip1));
        ip0 += is0;
        ip1 += is1;
    }
    *(flint*)op = f;
}

static npy_bool
npyflint_nonzero(void* data, void* arr) {
    flint f;
    memcpy(&f,data,sizeof(f));
    return flint_nonzero(f)?NPY_TRUE:NPY_FALSE;
}


static int
npyflint_fill(void* data_, npy_intp length, void* arr) {
    flint* data = (flint*)data_;
    flint delta = flint_subtract(data[1],data[0]);
    flint f = data[1];
    npy_intp i;
    for (i = 2; i < length; i++) {
        f = flint_add(f,delta);
        data[i] = f;
    }
    return 0;
}

static int
npyflint_fillwithscalar(void* buffer_, npy_intp length,
                           void* value, void* arr) {
    flint f = *(flint*)value;
    flint* buffer = (flint*)buffer_;
    npy_intp i;
    for (i = 0; i < length; i++) {
        buffer[i] = f;
    }
    return 0;
}

static PyArray_ArrFuncs npyflint_arrfuncs;

typedef struct { char c; flint f; } align_test;


PyArray_Descr npyflint_descr = {
        PyObject_HEAD_INIT(0)
        &PyFlint_Type,          /* typeobj */
        'V',                    /* kind */
        'r',                    /* type */
        '=',                    /* byteorder */
        /*
         * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
         * exceptions.  This isn't technically necessary,
         * since we're careful about thread safety, and hopefully future
         * versions of numpy will recognize that.
         */
        NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* hasobject */
        0,                      /* type_num */
        sizeof(flint),          /* elsize */
        offsetof(align_test,f), /* alignment */
        0,                      /* subarray */
        0,                      /* fields */
        0,                      /* names */
        &npyflint_arrfuncs,     /* f */
};


#define DEFINE_CAST(From,To,statement) \
    static void \
    npycast_##From##_##To(void* from_, void* to_, npy_intp n, void* fromarr, void* toarr) { \
        const From* from = (From*)from_; \
        To* to = (To*)to_; \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            From x = from[i]; \
            statement \
            to[i] = y; \
        } \
    }
#define DEFINE_INT_CAST(bits) \
    DEFINE_CAST(int##bits##_t,flint,flint y = make_flint_int(x);) \
    DEFINE_CAST(flint,int##bits##_t,int64_t z = flint_int(x); int##bits##_t y = z; if (y != z) set_overflow();)
DEFINE_INT_CAST(8)
DEFINE_INT_CAST(16)
DEFINE_INT_CAST(32)
DEFINE_INT_CAST(64)
DEFINE_CAST(flint,float,double y = flint_double(x);)
DEFINE_CAST(flint,double,double y = flint_double(x);)
DEFINE_CAST(npy_bool,flint,flint y = make_flint_int(x);)
DEFINE_CAST(flint,npy_bool,npy_bool y = flint_nonzero(x);)

#define BINARY_UFUNC(name,intype0,intype1,outtype,exp) \
    void name(char** args, npy_intp* dimensions, npy_intp* steps, void* data) { \
        npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions; \
        char *i0 = args[0], *i1 = args[1], *o = args[2]; \
        int k; \
        for (k = 0; k < n; k++) { \
            intype0 x = *(intype0*)i0; \
            intype1 y = *(intype1*)i1; \
            *(outtype*)o = exp; \
            i0 += is0; i1 += is1; o += os; \
        } \
    }
#define FLINT_BINARY_UFUNC(name,type,exp) BINARY_UFUNC(flint_ufunc_##name,flint,flint,type,exp)
FLINT_BINARY_UFUNC(add,flint,flint_add(x,y))
FLINT_BINARY_UFUNC(subtract,flint,flint_subtract(x,y))
FLINT_BINARY_UFUNC(multiply,flint,flint_multiply(x,y))
FLINT_BINARY_UFUNC(divide,flint,flint_divide(x,y))
FLINT_BINARY_UFUNC(remainder,flint,flint_remainder(x,y))
FLINT_BINARY_UFUNC(floor_divide,flint,make_flint_int(flint_floor(flint_divide(x,y))))
PyUFuncGenericFunction flint_ufunc_true_divide = flint_ufunc_divide;
FLINT_BINARY_UFUNC(minimum,flint,flint_lt(x,y)?x:y)
FLINT_BINARY_UFUNC(maximum,flint,flint_lt(x,y)?y:x)
FLINT_BINARY_UFUNC(equal,npy_bool,flint_eq(x,y))
FLINT_BINARY_UFUNC(not_equal,npy_bool,flint_ne(x,y))
FLINT_BINARY_UFUNC(less,npy_bool,flint_lt(x,y))
FLINT_BINARY_UFUNC(greater,npy_bool,flint_gt(x,y))
FLINT_BINARY_UFUNC(less_equal,npy_bool,flint_le(x,y))
FLINT_BINARY_UFUNC(greater_equal,npy_bool,flint_ge(x,y))

#define UNARY_UFUNC(name,type,exp) \
    void flint_ufunc_##name(char** args, npy_intp* dimensions, npy_intp* steps, void* data) { \
        npy_intp is = steps[0], os = steps[1], n = *dimensions; \
        char *i = args[0], *o = args[1]; \
        int k; \
        for (k = 0; k < n; k++) { \
            flint x = *(flint*)i; \
            *(type*)o = exp; \
            i += is; o += os; \
        } \
    }
UNARY_UFUNC(negative,flint,flint_negative(x))
UNARY_UFUNC(absolute,flint,flint_abs(x))
UNARY_UFUNC(floor,flint,make_flint_int(flint_floor(x)))
UNARY_UFUNC(ceil,flint,make_flint_int(flint_ceil(x)))
UNARY_UFUNC(trunc,flint,make_flint_int(flint_int(x)))
UNARY_UFUNC(square,flint,flint_multiply(x,x))
UNARY_UFUNC(rint,flint,make_flint_int(flint_rint(x)))
UNARY_UFUNC(sign,flint,make_flint_int(flint_sign(x)))
UNARY_UFUNC(reciprocal,flint,flint_inverse(x))

static NPY_INLINE void
flint_matrix_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    /* pointers to data for input and output arrays */
    char *ip1 = args[0];
    char *ip2 = args[1];
    char *op = args[2];

    /* lengths of core dimensions */
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    /* striding over core dimensions */
    npy_intp is1_m = steps[0];
    npy_intp is1_n = steps[1];
    npy_intp is2_n = steps[2];
    npy_intp is2_p = steps[3];
    npy_intp os_m = steps[4];
    npy_intp os_p = steps[5];

    /* core dimensions counters */
    npy_intp m, p;

    /* calculate dot product for each row/column vector pair */
    for (m = 0; m < dm; m++) {
        for (p = 0; p < dp; p++) {
            npyflint_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

            /* advance to next column of 2nd input array and output array */
            ip2 += is2_p;
            op  +=  os_p;
        }

        /* reset to first column of 2nd input array and output array */
        ip2 -= is2_p * p;
        op -= os_p * p;

        /* advance to next row of 1st input array and output array */
        ip1 += is1_m;
        op += os_m;
    }
}


static void
flint_gufunc_matrix_multiply(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func))
{
    /* outer dimensions counter */
    npy_intp N_;

    /* length of flattened outer dimensions */
    npy_intp dN = dimensions[0];

    /* striding over flattened outer dimensions for input and output arrays */
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];

    /* loop through outer dimensions, performing matrix multiply on core dimensions for each loop */
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
        flint_matrix_multiply(args, dimensions+1, steps+3);
    }
}



PyMethodDef module_methods[] = {
        {0} /* sentinel */
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "flint",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_flint(void) {
#else
PyMODINIT_FUNC initflint(void) {
#endif

    PyObject *m;

    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_umath();
    if (PyErr_Occurred()) {
        return NULL;
    }
    PyObject* numpy_str = PyUString_FromString("numpy");
    if (!numpy_str) {
        return NULL;
    }
    PyObject* numpy = PyImport_Import(numpy_str);
    Py_DECREF(numpy_str);
    if (!numpy) {
        return NULL;
    }

    /* Can't set this until we import numpy */
    PyFlint_Type.tp_base = &PyGenericArrType_Type;

    /* Initialize flint type object */
    if (PyType_Ready(&PyFlint_Type) < 0) {
        return NULL;
    }

    /* Initialize flint descriptor */
    PyArray_InitArrFuncs(&npyflint_arrfuncs);
    npyflint_arrfuncs.getitem = npyflint_getitem;
    npyflint_arrfuncs.setitem = npyflint_setitem;
    npyflint_arrfuncs.copyswapn = npyflint_copyswapn;
    npyflint_arrfuncs.copyswap = npyflint_copyswap;
    npyflint_arrfuncs.compare = npyflint_compare;
    npyflint_arrfuncs.argmin = npyflint_argmin;
    npyflint_arrfuncs.argmax = npyflint_argmax;
    npyflint_arrfuncs.dotfunc = npyflint_dot;
    npyflint_arrfuncs.nonzero = npyflint_nonzero;
    npyflint_arrfuncs.fill = npyflint_fill;
    npyflint_arrfuncs.fillwithscalar = npyflint_fillwithscalar;
    /* Left undefined: scanfunc, fromstr, sort, argsort */
    Py_TYPE(&npyflint_descr) = &PyArrayDescr_Type;
    int npy_flint = PyArray_RegisterDataType(&npyflint_descr);
    if (npy_flint<0) {
        return NULL;
    }

    /* Support dtype(flint) syntax */
    if (PyDict_SetItemString(PyFlint_Type.tp_dict,"dtype",(PyObject*)&npyflint_descr)<0) {
        return NULL;
    }

        /* Register casts to and from flint */
#define REGISTER_CAST(From,To,from_descr,to_typenum,safe) \
        PyArray_Descr* from_descr_##From##_##To = (from_descr); \
        if (PyArray_RegisterCastFunc(from_descr_##From##_##To,(to_typenum),npycast_##From##_##To)<0) { \
            return NULL; \
        } \
        if (safe && PyArray_RegisterCanCast(from_descr_##From##_##To,(to_typenum),NPY_NOSCALAR)<0) { \
            return NULL; \
        }
#define REGISTER_INT_CASTS(bits) \
        REGISTER_CAST(int##bits##_t,flint,PyArray_DescrFromType(NPY_INT##bits),npy_flint,1) \
        REGISTER_CAST(flint,int##bits##_t,&npyflint_descr,NPY_INT##bits,0)
    REGISTER_INT_CASTS(8)
    REGISTER_INT_CASTS(16)
    REGISTER_INT_CASTS(32)
    REGISTER_INT_CASTS(64)
    REGISTER_CAST(flint,float,&npyflint_descr,NPY_FLOAT,0)
    REGISTER_CAST(flint,double,&npyflint_descr,NPY_DOUBLE,1)
    REGISTER_CAST(npy_bool,flint,PyArray_DescrFromType(NPY_BOOL),npy_flint,1)
    REGISTER_CAST(flint,npy_bool,&npyflint_descr,NPY_BOOL,0)

        /* Register ufuncs */
#define REGISTER_UFUNC(name,...) { \
        PyUFuncObject* ufunc = (PyUFuncObject*)PyObject_GetAttrString(numpy,#name); \
        if (!ufunc) { \
            return NULL; \
        } \
        int _types[] = __VA_ARGS__; \
        if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
            PyErr_Format(PyExc_AssertionError,"ufunc %s takes %d arguments, our loop takes %ld",#name,ufunc->nargs,sizeof(_types)/sizeof(int)); \
            return NULL; \
        } \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_flint,flint_ufunc_##name,_types,0)<0) { \
            return NULL; \
        } \
    }
#define REGISTER_UFUNC_BINARY_FLINT(name) REGISTER_UFUNC(name,{npy_flint,npy_flint,npy_flint})
#define REGISTER_UFUNC_BINARY_COMPARE(name) REGISTER_UFUNC(name,{npy_flint,npy_flint,NPY_BOOL})
#define REGISTER_UFUNC_UNARY(name) REGISTER_UFUNC(name,{npy_flint,npy_flint})
    /* Binary */
    REGISTER_UFUNC_BINARY_FLINT(add)
    REGISTER_UFUNC_BINARY_FLINT(subtract)
    REGISTER_UFUNC_BINARY_FLINT(multiply)
    REGISTER_UFUNC_BINARY_FLINT(divide)
    REGISTER_UFUNC_BINARY_FLINT(remainder)
    REGISTER_UFUNC_BINARY_FLINT(true_divide)
    REGISTER_UFUNC_BINARY_FLINT(floor_divide)
    REGISTER_UFUNC_BINARY_FLINT(minimum)
    REGISTER_UFUNC_BINARY_FLINT(maximum)
    /* Comparisons */
    REGISTER_UFUNC_BINARY_COMPARE(equal)
    REGISTER_UFUNC_BINARY_COMPARE(not_equal)
    REGISTER_UFUNC_BINARY_COMPARE(less)
    REGISTER_UFUNC_BINARY_COMPARE(greater)
    REGISTER_UFUNC_BINARY_COMPARE(less_equal)
    REGISTER_UFUNC_BINARY_COMPARE(greater_equal)
    /* Unary */
    REGISTER_UFUNC_UNARY(negative)
    REGISTER_UFUNC_UNARY(absolute)
    REGISTER_UFUNC_UNARY(floor)
    REGISTER_UFUNC_UNARY(ceil)
    REGISTER_UFUNC_UNARY(trunc)
    REGISTER_UFUNC_UNARY(rint)
    REGISTER_UFUNC_UNARY(square)
    REGISTER_UFUNC_UNARY(reciprocal)
    REGISTER_UFUNC_UNARY(sign)

    /* Create module */
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("flint", module_methods);
#endif

    if (!m) {
        return NULL;
    }

    /* Add flint type */
    Py_INCREF(&PyFlint_Type);
    PyModule_AddObject(m,"flint",(PyObject*)&PyFlint_Type);

    /* Create matrix multiply generalized ufunc */
    PyObject* gufunc = PyUFunc_FromFuncAndDataAndSignature(0,0,0,0,2,1,PyUFunc_None,(char*)"matrix_multiply",(char*)"return result of multiplying two matrices of flints",0,"(m,n),(n,p)->(m,p)");
    if (!gufunc) {
        return NULL;
    }
    int types2[3] = {npy_flint,npy_flint,npy_flint};
    if (PyUFunc_RegisterLoopForType((PyUFuncObject*)gufunc,npy_flint,flint_gufunc_matrix_multiply,types2,0) < 0) {
        return NULL;
    }
    PyModule_AddObject(m,"matrix_multiply",(PyObject*)gufunc);

    /* Create numerator and denominator ufuncs */
#define NEW_UNARY_UFUNC(name,type,doc) { \
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,1,1,PyUFunc_None,(char*)#name,(char*)doc,0); \
        if (!ufunc) { \
            return NULL; \
        } \
        int types[2] = {npy_flint,type}; \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_flint,flint_ufunc_##name,types,0)<0) { \
            return NULL; \
        } \
        PyModule_AddObject(m,#name,(PyObject*)ufunc); \
    }
// NEW_UNARY_UFUNC(numerator,NPY_INT64,"flint number numerator");



    return m;
}

#ifdef __cplusplus
}
#endif