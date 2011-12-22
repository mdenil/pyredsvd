#cython: boundscheck=False
#cython: wraparound=False

import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from cpython cimport bool

cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass VectorXd:
        VectorXd()
        double *data()

    cdef cppclass MatrixXd:
        MatrixXd()
        double *data()

    cdef cppclass Map[T]:
        Map(double*, int, int)
        double operator()(int, int)

cdef extern from "redsvd.hpp" namespace "REDSVD":
    cdef cppclass RedSVD[Mat]:
        RedSVD(Mat&, int)

        MatrixXd& matrixU()
        VectorXd& singularValues()
        MatrixXd& matrixV()

cdef extern from "util.hpp" namespace "REDSVD":
    cdef cppclass SMatrixXd:
        SMatrixXd(int n, int m)

    cdef SMatrixXd *fill_sparse_matrix(SMatrixXd *A, int nnz, int *I_data, int *J_data, double *V_data)

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

cdef extern from "stdio.h":
    cdef void *memcpy (void *dest, void *src, size_t size)

cdef inline np.ndarray array_d(double *data, int n, int m):
    #cdef ndarray ary2 = PyArray_ZEROS(1, &size, 12, 0)
    cdef np.ndarray ary = np.zeros(shape=(n,m), dtype=np.float64)
    if data != NULL: memcpy(ary.data, data, n*m*sizeof(double))
    return ary

cdef inline np.ndarray vector_d(double *data, int size):
    cdef np.ndarray vec = np.zeros(size, dtype=np.float64)
    if data != NULL: memcpy(vec.data, data, size*sizeof(double))
    return vec
