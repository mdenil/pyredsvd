#cython: boundscheck=False
#cython: wraparound=False

import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from pyredsvd cimport *

# !! Important for using numpy's C api
np.import_array()

def dense_redsvd(np.ndarray[double, ndim=2, mode='c'] A, int rank):
    cdef double* A_data = <double*>A.data
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef Map[MatrixXd] *A_map = new Map[MatrixXd](A_data, n, m)

    # Create output matrices (and corresponding Eigen::Maps to
    # access their data)
    cdef np.ndarray U = np.ndarray(shape=(n, rank))
    cdef np.ndarray S = np.ndarray(shape=(rank,))
    cdef np.ndarray V = np.ndarray(shape=(m, rank))

    cdef double *U_data = <double*>U.data
    cdef double *S_data = <double*>S.data
    cdef double *V_data = <double*>V.data

    cdef Map[MatrixXd] *U_map = new Map[MatrixXd](U_data, n, rank)
    cdef Map[VectorXd] *S_map = new Map[VectorXd](S_data, rank, 1)
    cdef Map[MatrixXd] *V_map = new Map[MatrixXd](V_data, m, rank)

    cdef RedSVD[Map[MatrixXd]] *svd = new RedSVD[Map[MatrixXd]](
            deref(A_map),
            rank,
            U_map,
            S_map,
            V_map
            )

    del svd
    del A_map

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
    cdef double *values_data = <double*>values.data

    # Create and populate a sparse matrix as input for redsvd
    cdef SMatrixXd *A = new SMatrixXd(n, m)
    A = fill_sparse_matrix(A, nnz, I_data, J_data, values_data)

    # Create output matrices (and corresponding Eigen::Maps to
    # access their data)
    cdef np.ndarray U = np.ndarray(shape=(n, rank))
    cdef np.ndarray S = np.ndarray(shape=(rank,))
    cdef np.ndarray V = np.ndarray(shape=(m, rank))

    cdef double *U_data = <double*>U.data
    cdef double *S_data = <double*>S.data
    cdef double *V_data = <double*>V.data

    cdef Map[MatrixXd] *U_map = new Map[MatrixXd](U_data, n, rank)
    cdef Map[VectorXd] *S_map = new Map[VectorXd](S_data, rank, 1)
    cdef Map[MatrixXd] *V_map = new Map[MatrixXd](V_data, m, rank)

    # run redsvd
    cdef RedSVD[SMatrixXd] *svd = new RedSVD[SMatrixXd](
            deref(A),
            rank,
            U_map,
            S_map,
            V_map
            )

    # clean up
    del A
    del svd

    return U, S, V

# Saving this in case it's useful in the future.
#
#def give_me_a_matrix():
#    # http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
#    #
#    # How to get data back from numpy:
#    # http://groups.google.com/group/cython-users/browse_thread/thread/22dd22be38e7685f/4d73e90f7fd59c26
#
#    cdef np.npy_intp sz = 100
#    cdef object pyobj = np.PyArray_SimpleNew(1, &sz, np.NPY_FLOAT64)
#
#    cdef double *internal_data = <double*>np.PyArray_DATA(pyobj)
#
#    return <object>pyobj

