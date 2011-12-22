/*
 *  Copyright (c) 2011 Daisuke Okanohara
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1. Redistributions of source code must retain the above Copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above Copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of the authors nor the names of its contributors
 *      may be used to endorse or promote products derived from this
 *      software without specific prior written permission.
 */

#ifndef REDSVD_UTIL_HPP__
#define REDSVD_UTIL_HPP__

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

namespace REDSVD {

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SMatrixXd;

class Util {
public:
    static void sampleGaussianMat(Eigen::MatrixXd& x);
    static void processGramSchmidt(Eigen::MatrixXd& mat);
    static double getSec();

private:
    static void sampleTwoGaussian(double& f1, double& f2);
};


SMatrixXd *fill_sparse_matrix(SMatrixXd *A, int nnz, int *I, int *J, double *V);

}

#endif // REDSVD_UTIL_HPP_
