#cython: boundscheck=False
#cython: wraparound=False

import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref

cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd()

    cdef cppclass Map[T]:
        Map(double*, int, int)
        double operator()(int, int)

def eigenify(np.ndarray[double, ndim=2, mode='c'] A):
    cdef double* A_data = <double*>A.data
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef Map[MatrixXd] *B = new Map[MatrixXd](A_data, n, m)

    for i in range(n):
        for j in range(m):
            print deref(B)(i,j),
        print ""

    return 2


