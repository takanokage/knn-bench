
#include "test_faiss.h"

#include "performance.h"
#include "ref.h"
#include "report.h"

#include <iomanip>
#include <iostream>
using namespace std;

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <IndexFlat.h>
#include <gpu/GpuIndexFlat.h>
#include <gpu/GpuIndexIVFFlat.h>
#include <gpu/StandardGpuResources.h>

// ----------------------------------------------------------------------------
// test faiss using a flat index.
// ----------------------------------------------------------------------------
double test_faiss_flat(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const std::vector<float>& gt_distances,
    const std::vector<int>& gt_indices,
    const int& nb_iterations,
    const bool& validation)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances(testSize * K);
    vector<int> test_indices(testSize * K);

    float *xb = (float*)trainPoints.data();
    float *xq = (float*)testPoints.data();

    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexFlatL2 index_flat(&res, DIM);

    // add vectors to the index
    index_flat.add(trainSize, xb);

    // search xq
    long *I = new long[K * testSize];
    float *D = new float[K * testSize];

    Performance::Start();

    for (int i = 0; i < nb_iterations; i++)
        index_flat.search(testSize, xq, K, D, I);

    Performance::Stop();
    double elapsed_time = Performance::Duration() / nb_iterations;

    for(int i = 0; i < K * testSize; i++)
    {
        test_indices[i] = (int)I[i];
        test_distances[i] = (float)D[i];
    }

    delete [] I;
    delete [] D;

    // Compute accuracy
    float distance_acc = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow("faiss_flat_index",
        elapsed_time,
        nb_iterations,
        distance_acc,
        index_accuracy,
        validation);

    return elapsed_time;
}


// ----------------------------------------------------------------------------
// test faiss using an IVF index.
// ----------------------------------------------------------------------------
double test_faiss_ivf(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const std::vector<float>& gt_distances,
    const std::vector<int>& gt_indices,
    const int& nb_iterations,
    const bool& validation)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances(testSize * K);
    vector<int> test_indices(testSize * K);

    float *xb = (float*)trainPoints.data();
    float *xq = (float*)testPoints.data();

    faiss::gpu::StandardGpuResources res;

    int nlist = 100;
    faiss::gpu::GpuIndexIVFFlat index_ivf(&res, DIM, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search

    assert(!index_ivf.is_trained);
    index_ivf.train(trainSize, xb);
    assert(index_ivf.is_trained);
    index_ivf.add(trainSize, xb);  // add vectors to the index

    long *I = new long[K * testSize];
    float *D = new float[K * testSize];

    Performance::Start();

    for (int i = 0; i < nb_iterations; i++)
        index_ivf.search(testSize, xq, K, D, I);

    Performance::Stop();
    double elapsed_time = Performance::Duration() / nb_iterations;

    for(int i = 0; i < K * testSize; i++)
    {
        test_indices[i] = (int)I[i];
        test_distances[i] = (float)D[i];
    }

    delete [] I;
    delete [] D;

    // Compute accuracy
    float distance_acc = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow("faiss_ivf_index",
        elapsed_time,
        nb_iterations,
        distance_acc,
        index_accuracy,
        validation);

    return elapsed_time;
}
