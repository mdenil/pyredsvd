#cython: boundscheck=False
#cython: wraparound=False

import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from pyredsvd import *

# !! Important for using numpy's C api
np.import_array()

def give_me_a_matrix():
    # http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
    #
    # How to get data back from numpy:
    # http://groups.google.com/group/cython-users/browse_thread/thread/22dd22be38e7685f/4d73e90f7fd59c26

    cdef np.npy_intp sz = 100
    cdef object pyobj = np.PyArray_SimpleNew(1, &sz, np.NPY_FLOAT64)

    cdef double *internal_data = <double*>np.PyArray_DATA(pyobj)

    return <object>pyobj

def dense_redsvd(np.ndarray[double, ndim=2, mode='c'] A, int rank):
    cdef double* A_data = <double*>A.data
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef Map[MatrixXd] *B = new Map[MatrixXd](A_data, n, m)

    cdef RedSVD[Map[MatrixXd]] *svd = new RedSVD[Map[MatrixXd]](deref(B), rank)

    cdef double* data
    
    data = <double*>svd.matrixU().data()
    cdef np.ndarray U = array_d(data, n, rank)

    data = <double*>svd.singularValues().data()
    cdef np.ndarray S = vector_d(data, rank)

    data = <double*>svd.matrixV().data()
    cdef np.ndarray V = array_d(data, m, rank)

    del svd

    return U, S, V


def sparse_redsvd(
        np.ndarray[int, ndim=1, mode='c'] I,
        np.ndarray[int, ndim=1, mode='c'] J,
        np.ndarray[double, ndim=1, mode='c'] values,
        int rank,
        sort=True,
        ):

    # sort the indices so they're in the correct order for efficient
    # insertion into an eigen sparse matrix
    if sort:
        ind = np.lexsort((J,I))
        I = I[ind]
        J = J[ind]
        values = values[ind]

    # grab some metadata we need
    cdef int n = I.max() + 1
    cdef int m = J.max() + 1 # +1 because of zero indexing
    cdef int nnz = values.size
    cdef int *I_data = <int*>I.data
    cdef int *J_data = <int*>J.data
    cdef double *V_data = <double*>values.data

    # Create and populate a sparse matrix for redsvd
    cdef SMatrixXd *A = new SMatrixXd(n, m)
    A = fill_sparse_matrix(A, nnz, I_data, J_data, V_data)

    # run redsvd
    cdef RedSVD[SMatrixXd] *svd = new RedSVD[SMatrixXd](deref(A), rank)


    # Extract the result from redsvd into numpy arrays
    cdef double* data
    
    data = <double*>svd.matrixU().data()
    cdef np.ndarray U = array_d(data, n, rank)

    data = <double*>svd.singularValues().data()
    cdef np.ndarray S = vector_d(data, rank)

    data = <double*>svd.matrixV().data()
    cdef np.ndarray V = array_d(data, m, rank)

    # clean up
    del A
    del svd

    return U, S, V

############################
# Utility functions for making raw arrays into ndarrays
############################

cdef inline np.ndarray array_d(double *data, int n, int m):
    #cdef ndarray ary2 = PyArray_ZEROS(1, &size, 12, 0)
    cdef np.ndarray ary = np.zeros(shape=(n,m), dtype=np.float64)
    if data != NULL: memcpy(ary.data, data, n*m*sizeof(double))
    return ary

cdef inline np.ndarray vector_d(double *data, int size):
    cdef np.ndarray vec = np.zeros(size, dtype=np.float64)
    if data != NULL: memcpy(vec.data, data, size*sizeof(double))
    return vec
