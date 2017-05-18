#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"
#include "numpy/npy_3kcompat.h"

#include "flint.h"

typedef struct {
    PyObject_HEAD
    flint obval;
} PyFlintScalarObject;

PyMemberDef PyFlintArrType_members[] = {
    {"real", T_DOUBLE, offsetof(PyFlintScalarObject, obval.w), READONLY,
        "The real component of the flint"},
    {"w", T_DOUBLE, offsetof(PyFlintScalarObject, obval.w), READONLY,
        "The real component of the flint"},
    {"x", T_DOUBLE, offsetof(PyFlintScalarObject, obval.x), READONLY,
        "The first imaginary component of the flint"},
    {"y", T_DOUBLE, offsetof(PyFlintScalarObject, obval.y), READONLY,
        "The second imaginary component of the flint"},
    {"z", T_DOUBLE, offsetof(PyFlintScalarObject, obval.z), READONLY,
        "The third imaginary component of the flint"},
    {NULL}
};

static PyObject * 
PyFlintArrType_get_components(PyObject *self, void *closure)
{
    flint *q = &((PyFlintScalarObject *)self)->obval;
    PyObject *tuple = PyTuple_New(4);
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q->w));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q->x));
    PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(q->y));
    PyTuple_SET_ITEM(tuple, 3, PyFloat_FromDouble(q->z));
    return tuple;
}

static PyObject *
PyFlintArrType_get_imag(PyObject *self, void *closure)
{
    flint *q = &((PyFlintScalarObject *)self)->obval;
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q->x));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q->y));
    PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(q->z));
    return tuple;
}

PyGetSetDef PyFlintArrType_getset[] = {
    {"components", PyFlintArrType_get_components, NULL,
        "The components of the flint as a (w,x,y,z) tuple", NULL},
    {"imag", PyFlintArrType_get_imag, NULL,
        "The imaginary part of the flint as an (x,y,z) tuple", NULL},
    {NULL}
};

PyTypeObject PyFlintArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "flint.flint",                    /* tp_name*/
    sizeof(PyFlintScalarObject),           /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    PyFlintArrType_members,                /* tp_members */
    PyFlintArrType_getset,                 /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

static PyArray_ArrFuncs _PyFlint_ArrFuncs;
PyArray_Descr *flint_descr;

static PyObject *
QUATERNION_getitem(char *ip, PyArrayObject *ap)
{
    flint f;
    PyObject *tuple;

    if ((ap == NULL) || PyArray_ISBEHAVED_RO(ap)) {
        q = *((flint *)ip);
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->copyswap(&q.w, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.x, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.y, ip+16, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.z, ip+24, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    tuple = PyTuple_New(4);
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q.w));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q.x));
    PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(q.y));
    PyTuple_SET_ITEM(tuple, 3, PyFloat_FromDouble(q.z));

    return tuple;
}

static int QUATERNION_setitem(PyObject *op, char *ov, PyArrayObject *ap)
{
    flint f;

    if (PyArray_IsScalar(op, Flint)) {
        q = ((PyFlintScalarObject *)op)->obval;
    }
    else {
        q.w = PyFloat_AsDouble(PyTuple_GetItem(op, 0));
        q.x = PyFloat_AsDouble(PyTuple_GetItem(op, 1));
        q.y = PyFloat_AsDouble(PyTuple_GetItem(op, 2));
        q.z = PyFloat_AsDouble(PyTuple_GetItem(op, 3));
    }
    if (PyErr_Occurred()) {
        if (PySequence_Check(op)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError,
                    "setting an array element with a sequence.");
        }
        return -1;
    }
    if (ap == NULL || PyArray_ISBEHAVED(ap))
        *((flint *)ov)=q;
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->copyswap(ov, &q.w, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+8, &q.x, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+16, &q.y, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+24, &q.z, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    return 0;
}

static void
QUATERNION_copyswap(flint *dst, flint *src,
        int swap, void *NPY_UNUSED(arr))
{
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswapn(dst, sizeof(double), src, sizeof(double), 4, swap, NULL);
    Py_DECREF(descr);
}

static void
QUATERNION_copyswapn(flint *dst, npy_intp dstride,
        flint *src, npy_intp sstride,
        npy_intp n, int swap, void *NPY_UNUSED(arr))
{
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswapn(&dst->w, dstride, &src->w, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->x, dstride, &src->x, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->y, dstride, &src->y, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->z, dstride, &src->z, sstride, n, swap, NULL);
    Py_DECREF(descr);
}

static int
QUATERNION_compare (flint *pa, flint *pb, PyArrayObject *NPY_UNUSED(ap))
{
    flint a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;

    anan = flint_isnan(a);
    bnan = flint_isnan(b);

    if (anan) {
        ret = bnan ? 0 : -1;
    } else if (bnan) {
        ret = 1;
    } else if(flint_less(a, b)) {
        ret = -1;
    } else if(flint_less(b, a)) {
        ret = 1;
    } else {
        ret = 0;
    }

    return ret;
}

static int
QUATERNION_argmax(flint *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
    npy_intp i;
    flint mp = *ip;

    *max_ind = 0;

    if (flint_isnan(mp)) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for (i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if (!(flint_less_equal(*ip, mp))) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (flint_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool
QUATERNION_nonzero (char *ip, PyArrayObject *ap)
{
    flint f;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        q = *(flint *)ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->copyswap(&q.w, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.x, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.y, ip+16, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&q.z, ip+24, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !flint_equal(q, (flint) {0,0,0,0});
}

static void
QUATERNION_fillwithscalar(flint *buffer, npy_intp length, flint *value, void *NPY_UNUSED(ignored))
{
    npy_intp i;
    flint val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_QUATERNION(TYPE, type)                                       \
static void                                                                    \
TYPE ## _to_flint(type *ip, flint *op, npy_intp n,                   \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->w = (double)(*ip++);                                               \
        op->x = 0;                                                             \
        op->y = 0;                                                             \
        op->z = 0;                                                             \
    }                                                                          \
}

MAKE_T_TO_QUATERNION(FLOAT, npy_uint32);
MAKE_T_TO_QUATERNION(DOUBLE, npy_uint64);
MAKE_T_TO_QUATERNION(LONGDOUBLE, npy_longdouble);
MAKE_T_TO_QUATERNION(BOOL, npy_bool);
MAKE_T_TO_QUATERNION(BYTE, npy_byte);
MAKE_T_TO_QUATERNION(UBYTE, npy_ubyte);
MAKE_T_TO_QUATERNION(SHORT, npy_short);
MAKE_T_TO_QUATERNION(USHORT, npy_ushort);
MAKE_T_TO_QUATERNION(INT, npy_int);
MAKE_T_TO_QUATERNION(UINT, npy_uint);
MAKE_T_TO_QUATERNION(LONG, npy_long);
MAKE_T_TO_QUATERNION(ULONG, npy_ulong);
MAKE_T_TO_QUATERNION(LONGLONG, npy_longlong);
MAKE_T_TO_QUATERNION(ULONGLONG, npy_ulonglong);

#define MAKE_CT_TO_QUATERNION(TYPE, type)                                      \
static void                                                                    \
TYPE ## _to_flint(type *ip, flint *op, npy_intp n,                   \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->w = (double)(*ip++);                                               \
        op->x = (double)(*ip++);                                               \
        op->y = 0;                                                             \
        op->z = 0;                                                             \
    }                                                                          \
}

MAKE_CT_TO_QUATERNION(CFLOAT, npy_uint32);
MAKE_CT_TO_QUATERNION(CDOUBLE, npy_uint64);
MAKE_CT_TO_QUATERNION(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject *
flint_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    flint f;

    if (!PyArg_ParseTuple(args, "dddd", &q.w, &q.x, &q.y, &q.z))
        return NULL;

    return PyArray_Scalar(&q, flint_descr, NULL);
}

static PyObject *
gentype_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long
flint_arrtype_hash(PyObject *o)
{
    flint f = ((PyFlintScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashDouble(q.w);
    value = (10000004 * value) ^ _Py_HashDouble(q.x);
    value = (10000004 * value) ^ _Py_HashDouble(q.y);
    value = (10000004 * value) ^ _Py_HashDouble(q.z);
    if (value == -1)
        value = -2;
    return value;
}

static PyObject *
flint_arrtype_repr(PyObject *o)
{
    char str[128];
    flint f = ((PyFlintScalarObject *)o)->obval;
    sprintf(str, "flint(%g, %g, %g, %g)", q.w, q.x, q.y, q.z);
    return PyUString_FromString(str);
}

static PyObject *
flint_arrtype_str(PyObject *o)
{
    char str[128];
    flint f = ((PyFlintScalarObject *)o)->obval;
    sprintf(str, "flint(%g, %g, %g, %g)", q.w, q.x, q.y, q.z);
    return PyString_FromString(str);
}

static PyMethodDef FlintMethods[] = {
    {NULL, NULL, 0, NULL}
};

#define UNARY_UFUNC(name, ret_type)\
static void \
flint_##name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){\
        const flint in1 = *(flint *)ip1;\
        *((ret_type *)op1) = flint_##name(in1);};}

UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(negative, flint)

#define BINARY_GEN_UFUNC(name, func_name, arg_type, ret_type)\
static void \
flint_##func_name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){\
        const flint in1 = *(flint *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = flint_##func_name(in1, in2);};};

#define BINARY_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name, flint, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name##_scalar, npy_double, ret_type)

BINARY_UFUNC(add, flint)
BINARY_UFUNC(subtract, flint)
BINARY_UFUNC(multiply, flint)
BINARY_UFUNC(divide, flint)
BINARY_UFUNC(power, flint)
BINARY_UFUNC(copysign, flint)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)

BINARY_SCALAR_UFUNC(multiply, flint)
BINARY_SCALAR_UFUNC(divide, flint)
BINARY_SCALAR_UFUNC(power, flint)

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_flint",
    NULL,
    -1,
    FlintMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_numpy_flint(void) {
#else
PyMODINIT_FUNC initnumpy_flint(void) {
#endif

    PyObject *m;
    int flintNum;
    PyObject* numpy = PyImport_ImportModule("numpy");
    PyObject* numpy_dict = PyModule_GetDict(numpy);
    int arg_types[3];

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("numpy_flint", FlintMethods);
#endif

    if (!m) {
        return NULL;
    }

    /* Make sure NumPy is initialized */
    import_array();
    import_umath();

    /* Register the flint array scalar type */
#if defined(NPY_PY3K)
    PyFlintArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyFlintArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyFlintArrType_Type.tp_new = flint_arrtype_new;
    PyFlintArrType_Type.tp_richcompare = gentype_richcompare;
    PyFlintArrType_Type.tp_hash = flint_arrtype_hash;
    PyFlintArrType_Type.tp_repr = flint_arrtype_repr;
    PyFlintArrType_Type.tp_str = flint_arrtype_str;
    PyFlintArrType_Type.tp_base = &PyGenericArrType_Type;
    if (PyType_Ready(&PyFlintArrType_Type) < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyFlintArrType_Type");
        return NULL;
    }

    /* The array functions */
    PyArray_InitArrFuncs(&_PyFlint_ArrFuncs);
    _PyFlint_ArrFuncs.getitem = (PyArray_GetItemFunc*)QUATERNION_getitem;
    _PyFlint_ArrFuncs.setitem = (PyArray_SetItemFunc*)QUATERNION_setitem;
    _PyFlint_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)QUATERNION_copyswap;
    _PyFlint_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)QUATERNION_copyswapn;
    _PyFlint_ArrFuncs.compare = (PyArray_CompareFunc*)QUATERNION_compare;
    _PyFlint_ArrFuncs.argmax = (PyArray_ArgFunc*)QUATERNION_argmax;
    _PyFlint_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)QUATERNION_nonzero;
    _PyFlint_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)QUATERNION_fillwithscalar;

    /* The flint array descr */
    flint_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    flint_descr->typeobj = &PyFlintArrType_Type;
    flint_descr->kind = 'q';
    flint_descr->type = 'j';
    flint_descr->byteorder = '=';
    flint_descr->type_num = 0; /* assigned at registration */
    flint_descr->elsize = 8*4;
    flint_descr->alignment = 8;
    flint_descr->subarray = NULL;
    flint_descr->fields = NULL;
    flint_descr->names = NULL;
    flint_descr->f = &_PyFlint_ArrFuncs;

    Py_INCREF(&PyFlintArrType_Type);
    flintNum = PyArray_RegisterDataType(flint_descr);

    if (flintNum < 0)
        return NULL;

    register_cast_function(NPY_BOOL, flintNum, (PyArray_VectorUnaryFunc*)BOOL_to_flint);
    register_cast_function(NPY_BYTE, flintNum, (PyArray_VectorUnaryFunc*)BYTE_to_flint);
    register_cast_function(NPY_UBYTE, flintNum, (PyArray_VectorUnaryFunc*)UBYTE_to_flint);
    register_cast_function(NPY_SHORT, flintNum, (PyArray_VectorUnaryFunc*)SHORT_to_flint);
    register_cast_function(NPY_USHORT, flintNum, (PyArray_VectorUnaryFunc*)USHORT_to_flint);
    register_cast_function(NPY_INT, flintNum, (PyArray_VectorUnaryFunc*)INT_to_flint);
    register_cast_function(NPY_UINT, flintNum, (PyArray_VectorUnaryFunc*)UINT_to_flint);
    register_cast_function(NPY_LONG, flintNum, (PyArray_VectorUnaryFunc*)LONG_to_flint);
    register_cast_function(NPY_ULONG, flintNum, (PyArray_VectorUnaryFunc*)ULONG_to_flint);
    register_cast_function(NPY_LONGLONG, flintNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_flint);
    register_cast_function(NPY_ULONGLONG, flintNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_flint);
    register_cast_function(NPY_FLOAT, flintNum, (PyArray_VectorUnaryFunc*)FLOAT_to_flint);
    register_cast_function(NPY_DOUBLE, flintNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_flint);
    register_cast_function(NPY_LONGDOUBLE, flintNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_flint);
    register_cast_function(NPY_CFLOAT, flintNum, (PyArray_VectorUnaryFunc*)CFLOAT_to_flint);
    register_cast_function(NPY_CDOUBLE, flintNum, (PyArray_VectorUnaryFunc*)CDOUBLE_to_flint);
    register_cast_function(NPY_CLONGDOUBLE, flintNum, (PyArray_VectorUnaryFunc*)CLONGDOUBLE_to_flint);

#define REGISTER_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            flint_descr->type_num, flint_##name##_ufunc, arg_types, NULL)

#define REGISTER_SCALAR_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            flint_descr->type_num, flint_##name##_scalar_ufunc, arg_types, NULL)

    /* quat -> bool */
    arg_types[0] = flint_descr->type_num;
    arg_types[1] = NPY_BOOL;

    REGISTER_UFUNC(isnan);
    REGISTER_UFUNC(isinf);
    REGISTER_UFUNC(isfinite);
    /* quat -> double */
    arg_types[1] = NPY_DOUBLE;

    REGISTER_UFUNC(absolute);

    /* quat -> quat */
    arg_types[1] = flint_descr->type_num;

    REGISTER_UFUNC(negative);

    /* quat, quat -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC(equal);
    REGISTER_UFUNC(not_equal);
    REGISTER_UFUNC(less);
    REGISTER_UFUNC(less_equal);

    /* quat, double -> quat */

    arg_types[1] = NPY_DOUBLE;
    arg_types[2] = flint_descr->type_num;

    REGISTER_SCALAR_UFUNC(multiply);
    REGISTER_SCALAR_UFUNC(divide);
    REGISTER_SCALAR_UFUNC(power);

    /* quat, quat -> quat */

    arg_types[1] = flint_descr->type_num;

    REGISTER_UFUNC(add);
    REGISTER_UFUNC(subtract);
    REGISTER_UFUNC(multiply);
    REGISTER_UFUNC(divide);
    REGISTER_UFUNC(power);


    PyModule_AddObject(m, "flint", (PyObject *)&PyFlintArrType_Type);

    return m;
}
