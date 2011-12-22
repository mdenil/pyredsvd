/*
 *  Copyright (c) 2010 Daisuke Okanohara
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

#include <iostream>
#include <sys/time.h>

#include "util.hpp"

using namespace std;
using namespace Eigen;

namespace REDSVD {

const double SVD_EPS = 0.0001f;

double Util::getSec() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

void Util::sampleTwoGaussian(double& f1, double& f2) {
    double v1 = (double)(rand() + 1.f) / ((double)RAND_MAX+2.f);
    double v2 = (double)(rand() + 1.f) / ((double)RAND_MAX+2.f);
    double len = sqrt(-2.f * log(v1));
    f1 = len * cos(2.f * M_PI * v2);
    f2 = len * sin(2.f * M_PI * v2);
}

void Util::sampleGaussianMat(MatrixXd& mat) {
    for (int i = 0; i < mat.rows(); ++i) {
        int j = 0;
        for ( ; j+1 < mat.cols(); j += 2) {
            double f1, f2;
            sampleTwoGaussian(f1, f2);
            mat(i,j  ) = f1;
            mat(i,j+1) = f2;
        }
        for (; j < mat.cols(); j ++) {
            double f1, f2;
            sampleTwoGaussian(f1, f2);
            mat(i, j)  = f1;
        }
    }
}


void Util::processGramSchmidt(MatrixXd& mat) {
    for (int i = 0; i < mat.cols(); ++i) {
        for (int j = 0; j < i; ++j) {
            double r = mat.col(i).dot(mat.col(j));
            mat.col(i) -= r * mat.col(j);
        }
        double norm = mat.col(i).norm();
        if (norm < SVD_EPS) {
            for (int k = i; k < mat.cols(); ++k) {
                mat.col(k).setZero();
            }
            return;
        }
        mat.col(i) *= (1.f / norm);
    }
}

/**
 * Fills a sparse matrix from I,J,V data.
 *
 * This function assumes the data are sorted in the correct order
 * for efficient insertion.  Horrible things will happen if this is
 * not the case.
 */
SMatrixXd *fill_sparse_matrix(SMatrixXd *A, int nnz, int *I, int *J, double *V)
{
    int prev_i = -1;
    A->reserve(nnz);
    for (int k=0; k<nnz; ++k) {
        int i = I[k];
        int j = J[k];

        while (i != prev_i) {
            // yes this really does need to be called for
            // each possible i
            A->startVec(++prev_i);
        }
        A->insertBackByOuterInner(i,j) = V[k];
    }
    A->finalize();

    return A;
}

}
