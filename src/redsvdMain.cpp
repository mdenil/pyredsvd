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

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "redsvd.hpp"

namespace REDSVD {

#define ENABLE_BENCHMARK

template<class Mat, class RetMat>
void fileProcess(const std::string& inputFileName,
                 const std::string& outputFileName,
                 int rank) {
    Mat A;
    //readMatrix(inputFileName.c_str(), A);

#ifdef ENABLE_BENCHMARK
    double startSec = Util::getSec();
    std::cout << Util::getSec() - startSec << " sec." <<std:: endl;
    std::cout << "rows:\t" << A.rows() << std::endl
              << "cols:\t" << A.cols() << std::endl
              << "rank:\t" << rank  << std::endl;

    startSec = Util::getSec();
#endif

    RetMat retMat(A, rank);

#ifdef ENABLE_BENCHMARK
    std::cout << Util::getSec() - startSec << " sec." << std::endl
              << "finished." << std::endl;

#endif

    //writeMatrix(outputFileName, retMat);
}

}

using namespace std;

int main(int argc, char* argv[]) {

    string method = "SVD";
    string input = "fake.in";
    string output = "fake.out";
    int rank = 10;
    bool isInputSparse = false;

    try {
        if (method == "SVD") {
            if (isInputSparse) {
                REDSVD::fileProcess<REDSVD::SMatrixXf, REDSVD::RedSVD>(input, output, rank);
            } else {
                REDSVD::fileProcess<Eigen::MatrixXf, REDSVD::RedSVD>(input, output, rank);
            }
        } else if (method == "PCA") {
            if (isInputSparse) {
                REDSVD::fileProcess<REDSVD::SMatrixXf, REDSVD::RedPCA>(input, output, rank);
            } else {
                REDSVD::fileProcess<Eigen::MatrixXf, REDSVD::RedPCA>(input, output, rank);
            }
        } else if (method == "SymEigen") {
            if (isInputSparse) {
                REDSVD::fileProcess<REDSVD::SMatrixXf, REDSVD::RedSymEigen>(input, output, rank);
            } else {
                REDSVD::fileProcess<Eigen::MatrixXf, REDSVD::RedSymEigen>(input, output, rank);
            }
        } else {
            cerr << "unknown method:" << method << endl;
            return -1;
        }
    } catch (const string& error) {
        cerr << "Error: " << error << endl;
        return -1;
    }
    return 0;
}
