
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
        RedSVD(Mat&,
                int,
                Map[MatrixXd] *,
                Map[VectorXd] *,
                Map[MatrixXd] *
                )

        MatrixXd& matrixU()
        VectorXd& singularValues()
        MatrixXd& matrixV()

cdef extern from "util.hpp" namespace "REDSVD":
    cdef cppclass SMatrixXd:
        SMatrixXd(int n, int m)

    cdef SMatrixXd *fill_sparse_matrix(SMatrixXd *A, int nnz, int *I_data, int *J_data, double *V_data)
