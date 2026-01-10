#ifndef CPPCOPY
#define CPPCOPY

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

/*
 * Copy n-by-m slice from slice_ptr to dst.
 *
 * Copied straight from scipy.
 */
template <typename T>
void copy_slice_F_to_C(T *dst, const T *slice_ptr, const npy_intp n, const npy_intp m,
                       const npy_intp s2, const npy_intp s1) {
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            dst[i * m + j] = *(slice_ptr + (i * s2 / sizeof(T)) + (j * s1 / sizeof(T)));
        }
    }
}

/*
 * Copy n-by-m slice from slice_ptr to dst.
 *
 * Copied straight from scipy.
 */
template<typename T>
void copy_slice(T* dst, const T* slice_ptr, const npy_intp n, const npy_intp m, const npy_intp s2, const npy_intp s1) {
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            dst[i * m + j] = *(slice_ptr + (i*s2/sizeof(T)) + (j*s1/sizeof(T)));
        }
    }
}

/*
 * Copied straight from scipy.
 */
template <typename T>
inline void swap_cf(T *src, T *dst, const Py_ssize_t r, const Py_ssize_t c, const Py_ssize_t n) {
    Py_ssize_t i, j, ith_row, r2, c2;
    T *bb = dst;
    T *aa = src;
    if ((r < 16) && (c < 16)) {
        for (j = 0; j < c; j++) {
            ith_row = 0;
            for (i = 0; i < r; i++) {
                bb[ith_row] = aa[i];
                ith_row += n;
            }
            aa += n;
            bb += 1;
        }
    } else {
        // If tall
        if (r > c) {
            r2 = r / 2;
            swap_cf(src, dst, r2, c, n);
            swap_cf(src + r2, dst + (r2)*n, r - r2, c, n);
        } else { // Nope
            c2 = c / 2;
            swap_cf(src, dst, r, c2, n);
            swap_cf(src + (c2)*n, dst + c2, r, c - c2, n);
        }
    }
}

/*
 * Copy n-by-m C-order slice from slice_ptr to dst in F-order.
 *
 * Copied straight from scipy.
 */
template <typename T>
void copy_slice_F(T *dst, const T *slice_ptr, const npy_intp n, const npy_intp m, const npy_intp s2,
                  const npy_intp s1) {

    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            dst[i + j * n] =
                *(slice_ptr + (i * s2 / sizeof(T)) + (j * s1 / sizeof(T))); // == src[i*m + j]
        }
    }
}

template <typename T>
inline void double_copy_inner(PyArrayObject *ap_Am, T *res) {
    T *Am_data = (T *)PyArray_DATA(ap_Am);
    npy_intp ndim = PyArray_NDIM(ap_Am);
    npy_intp *shape = PyArray_SHAPE(ap_Am);
    npy_intp *strides = PyArray_STRIDES(ap_Am);

    npy_intp outer_size = 1;
    if (ndim > 2) {
        for (int i = 0; i < ndim - 2; i++) {
            outer_size *= shape[i];
        }
    }

    npy_intp m = shape[ndim - 2], n = shape[ndim - 1];

    T *scratch = (T *)malloc(m * n * sizeof(T));
    if (scratch == NULL) {
        PyErr_NoMemory();
        // Would lead to a memory leak on error, but a simple toy script.
    }

    // Actual copying, looping mechanism from scipy.
    // No need to compute two separate pointers, the
    // shapes of the two matrices are identical.
    for (npy_intp idx = 0; idx < outer_size; idx++) {

        npy_intp offset = 0;
        npy_intp temp_idx = idx;

        for (int i = ndim - 3; i >= 0; i--) {
            offset += (temp_idx % shape[i]) * strides[i];
            temp_idx /= shape[i];
        }

        T *slice_ptr = (T *)(Am_data + (offset / sizeof(T)));
        T *res_ptr = (T *)(res + (offset / sizeof(T)));

        copy_slice(scratch, slice_ptr, m, n, strides[ndim - 2], strides[ndim - 1]);
        swap_cf(scratch, res_ptr, m, n, n);
    }
}

template <typename T>
inline void single_copy_inner(PyArrayObject *ap_Am, T *res) {
    T *Am_data = (T *)PyArray_DATA(ap_Am);
    npy_intp ndim = PyArray_NDIM(ap_Am);
    npy_intp *shape = PyArray_SHAPE(ap_Am);
    npy_intp *strides = PyArray_STRIDES(ap_Am);

    npy_intp outer_size = 1;
    if (ndim > 2) {
        for (int i = 0; i < ndim - 2; i++) {
            outer_size *= shape[i];
        }
    }

    npy_intp m = shape[ndim - 2], n = shape[ndim - 1];

    // Actual copying, looping mechanism from scipy.
    // No need to compute two separate pointers, the
    // shapes of the two matrices is identical.
    for (npy_intp idx = 0; idx < outer_size; idx++) {

        npy_intp offset = 0;
        npy_intp temp_idx = idx;

        for (int i = ndim - 3; i >= 0; i--) {
            offset += (temp_idx % shape[i]) * strides[i];
            temp_idx /= shape[i];
        }

        T *slice_ptr = (T *)(Am_data + (offset / sizeof(T)));
        T *res_ptr = (T *)(res + (offset / sizeof(T)));

        copy_slice_F(res_ptr, slice_ptr, m, n, strides[ndim - 2], strides[ndim - 1]);
    }
}

static PyObject *double_copy(PyObject *Py_UNUSED(dummy), PyObject *args) {
    PyArrayObject *ap_Am = NULL;
    PyArrayObject *ap_res = NULL;

    if (!PyArg_ParseTuple(args, ("O!"), &PyArray_Type, (PyObject **)&ap_Am)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse arguments.");
        return NULL;
    }

    int typenum = PyArray_TYPE(ap_Am);
    bool dtype_ok = (typenum == NPY_FLOAT32) || (typenum == NPY_FLOAT64)
                 || (typenum == NPY_COMPLEX64) || (typenum == NPY_COMPLEX128);

    if (!PyArray_ISALIGNED(ap_Am)) {
        PyErr_SetString(PyExc_TypeError, "Expected an aligned array.");
        return NULL;
    }

    npy_intp ndim = PyArray_NDIM(ap_Am);
    npy_intp *shape = PyArray_SHAPE(ap_Am);

    if (ndim < 2) {
        PyErr_SetString(PyExc_ValueError, "Input array should at least be 2D.");
        return NULL;
    }

    ap_res = (PyArrayObject *)PyArray_SimpleNew(ndim, shape, typenum);
    if (!ap_res) {
        PyErr_NoMemory();
        return NULL;
    }

    void *res_data = PyArray_DATA(ap_res);

    switch (typenum) {
        case (NPY_FLOAT32):
            double_copy_inner<float>(ap_Am, (float *)res_data);
            break;

        case (NPY_FLOAT64):
            double_copy_inner<double>(ap_Am, (double *)res_data);
            break;

        /*
        case (NPY_COMPLEX64):
            double_copy_inner<npy_complex64>(ap_Am, (npy_complex64 *)res_data);
            break;

        case (NPY_COMPLEX128):
            double_copy_inner<npy_complex128>(ap_Am, (npy_complex128 *)res_data);
            break;
        */
    }

    return Py_BuildValue("N", PyArray_Return(ap_res));
}

static PyObject *single_copy(PyObject *Py_UNUSED(dummy), PyObject *args) {
    // General bookkeeping
    PyArrayObject *ap_Am = NULL;
    PyArrayObject *ap_res = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, (PyObject **)&ap_Am)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse arguments.");
        return NULL;
    }

    int typenum = PyArray_TYPE(ap_Am);
    bool dtype_ok = (typenum == NPY_FLOAT32) || (typenum == NPY_FLOAT64)
                 || (typenum == NPY_COMPLEX64) || (typenum == NPY_COMPLEX128);

    if (!dtype_ok || !PyArray_ISALIGNED(ap_Am)) {
        PyErr_SetString(PyExc_TypeError, "Expected an aligned array.");
        return NULL;
    }

    npy_intp ndim = PyArray_NDIM(ap_Am);
    npy_intp *shape = PyArray_SHAPE(ap_Am);
    npy_intp *strides = PyArray_STRIDES(ap_Am);

    if (ndim < 2) {
        PyErr_SetString(PyExc_ValueError, "Input array should at least be 2D.");
        return NULL;
    }

    ap_res = (PyArrayObject *)PyArray_SimpleNew(ndim, shape, typenum);
    if (!ap_res) {
        PyErr_NoMemory();
        return NULL;
    }

    void *res_data = PyArray_DATA(ap_res);

    switch (typenum) {
        case (NPY_FLOAT32):
            single_copy_inner<float>(ap_Am, (float *)res_data);
            break;

        case (NPY_FLOAT64):
            single_copy_inner<double>(ap_Am, (double *)res_data);
            break;

        /*
        case (NPY_COMPLEX64):
            single_copy_inner<npy_complex64>(ap_Am, (npy_complex64 *)res_data);
            break;

        case (NPY_COMPLEX128):
            single_copy_inner<npy_complex128>(ap_Am, (npy_complex128 *)res_data);
            break;
        */
    }

    return Py_BuildValue("N", PyArray_Return(ap_res));
}

static char doc_single_copy[] = ("Do a conversion from C to F ordering in one go.");
static char doc_double_copy[] =
    ("Do a conversion from C to F ordering by first copying and only then "
     "swapping ordering.");

static struct PyMethodDef cppcopy_module_methods[] = {
    {"single_copy", single_copy, METH_VARARGS, doc_single_copy},
    {"double_copy", double_copy, METH_VARARGS, doc_double_copy},
    {NULL, 0, NULL, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "cppcopy", NULL, -1, cppcopy_module_methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_cppcopy(void) {
    import_array();
    return PyModule_Create(&moduledef);
}

#endif // CPPCOPY
