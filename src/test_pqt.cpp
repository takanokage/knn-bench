
#include "test_pqt.h"

#include <stdlib.h>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda.h>

#include "VectorQuantization.hh"
#include "ProductQuantization.hh"
#include "ProQuantization.hh"
#include "ProTree.hh"
#include "PerturbationProTree.hh"
using namespace pqt;

#include "ref.h"
#include "report.h"

#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
using namespace std;

double test_pqt(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const std::vector<float>& gt_distances,
    const std::vector<int>& gt_indices,
    const char* const name,
    const int& nb_iterations,
    const bool& validation)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

	int bGPU = 0;
	cudaSetDevice(bGPU);

	uint N = trainSize;
	uint QN = testSize;

	int p = 1;

	float* M = (float*)trainPoints.data();
	float* Q = (float*)testPoints.data();

	float *Md, *Qd;
	float *Distd;

	cudaMalloc(&Md, N * DIM * sizeof(float));

	QN = K;

	cudaMalloc(&Qd, QN * DIM * sizeof(float));
	cudaMalloc(&Distd, N * QN * sizeof(float));

	cudaMemcpy(Md, M, N * DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, Q, QN * DIM * sizeof(float), cudaMemcpyHostToDevice);

	int k = 16;

	PerturbationProTree ppt(DIM, p, p);

	vector<float> dVec(QN * N);
    float *d =  dVec.data();

	ppt.calcDist(Distd, Md, Qd, N, QN, DIM, 1);

    /*/// v0 - runtime error, possibly fixable in CMakeLists.txt
    // terminate called after throwing an instance of 'thrust::system::system_error'
    // what():  radix_sort: failed on 1st step: invalid device function
	for (int i = 0; i < QN; i++)
		ppt.parallelSort(Distd, (i*N), (i+1)*N);

	cudaMemcpy(d, Distd, QN * N * sizeof(float), cudaMemcpyDeviceToHost);
    /*/// v1
	cudaMemcpy(d, Distd, QN * N * sizeof(float), cudaMemcpyDeviceToHost);

	// sort each vector independently
	for (int i = 0; i < QN; i++)
		std::sort(dVec.begin() + (i * N), dVec.begin() + ((i+1) * N));
    //*///

	cudaFree(Md);
	cudaFree(Distd);
	cudaFree(Qd);
}
