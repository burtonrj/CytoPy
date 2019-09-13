#include <Python.h>
#include <numpy/arrayobject.h>
#include "logicle.h"

static PyObject *wrap_logicle_scale(PyObject *self, PyObject *args) {
    double t, w, m, a;
    PyObject *x;

    // parse the input args tuple
    if (!PyArg_ParseTuple(args, "ddddO", &t, &w, &m, &a, &x)) {
        return NULL;
    }

    // read the numpy array
    PyObject *x_array = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY);

    // throw exception if the array doesn't exist
    if (x_array == NULL) {
        Py_XDECREF(x_array);
        return NULL;
    }

    // get length of input array
    int n = (int)PyArray_DIM(x_array, 0);

    // get pointers to the data as C-type
    double *xc    = (double*)PyArray_DATA(x_array);

    // now we can call our function!
    logicle_scale(t, w, m, a, xc, n);

    return x_array;
}

static PyObject *wrap_hyperlog_scale(PyObject *self, PyObject *args) {
    double t, w, m, a;
    PyObject *x;

    // parse the input args tuple
    if (!PyArg_ParseTuple(args, "ddddO", &t, &w, &m, &a, &x)) {
        return NULL;
    }

    // read the numpy array
    PyObject *x_array = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY);

    // throw exception if the array doesn't exist
    if (x_array == NULL) {
        Py_XDECREF(x_array);
        return NULL;
    }

    // get length of input array
    int n = (int)PyArray_DIM(x_array, 0);

    // get pointers to the data as C-type
    double *xc    = (double*)PyArray_DATA(x_array);

    // now we can call our function!
    hyperlog_scale(t, w, m, a, xc, n);

    return x_array;
}

static PyMethodDef module_methods[] = {
    {"logicle_scale", wrap_logicle_scale, METH_VARARGS, NULL},
    {"hyperlog_scale", wrap_hyperlog_scale, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef logicledef = {
        PyModuleDef_HEAD_INIT,
        "logicle_c",
        NULL,
        -1,
        module_methods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_logicle_c(void) {
    PyObject *m = PyModule_Create(&logicledef);
#else
PyMODINIT_FUNC initlogicle_c(void) {
    PyObject *m = Py_InitModule3("logicle_c", module_methods, NULL);
#endif

    if (m == NULL) {
        return NULL;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}