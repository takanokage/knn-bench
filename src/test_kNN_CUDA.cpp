
#include "test_kNN_CUDA.h"

#include "performance.h"
#include "ref.h"
#include "report.h"

#include <cmath>

#include <iomanip>
#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
double test_kNN_CUDA(
    const vector<float>& trainPoints,
    const vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const vector<float>& gt_distances,
    const vector<int>& gt_indices,
    const int& nb_iterations,
    const bool& validation)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    // for some reason kNN-CUDA works with DIM & K as leading dimensions
    // rows: DIM & K
    // cols: trainSize & testSize
    vector<float> trainPointsTr = Transpose(trainPoints, trainSize, DIM);
    vector<float> testPointsTr = Transpose(testPoints, testSize, DIM);
    vector<float> gt_distancesTr = Transpose(gt_distances, testSize, K);
    vector<int> gt_indicesTr = Transpose(gt_indices, testSize, K);

    test(trainPointsTr, testPointsTr, DIM, K,
        gt_distancesTr, gt_indicesTr,
        &knn_cuda_global,  "knn_cuda_global",
        100, validation);

    test(trainPointsTr, testPointsTr, DIM, K,
        gt_distancesTr, gt_indicesTr,
        &knn_cuda_texture, "knn_cuda_texture",
        100, validation);

    test(trainPointsTr, testPointsTr, DIM, K,
        gt_distancesTr, gt_indicesTr,
        &knn_cublas,       "knn_cublas",
        100, validation);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
double test(
    const vector<float>& trainPoints,
    const vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const vector<float>& gt_distances,
    const vector<int>& gt_indices,
    bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
    const char* const name,
    const int& nb_iterations,
    const bool& validation)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances(testSize * K);
    vector<int> test_indices(testSize * K);

    Performance::Start();

    // Compute K-NN several times
    for (int i = 0; i < nb_iterations; i++)
        knn(&trainPoints[0],
            trainSize,
            &testPoints[0],
            testSize,
            DIM,
            K,
            &test_distances[0],
            &test_indices[0]);

    Performance::Stop();
    double elapsed_time = Performance::Duration() / nb_iterations;

    // Compute accuracy
    float distance_acc   = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow(name,
        elapsed_time,
        nb_iterations,
        distance_acc,
        index_accuracy,
        validation);

    return elapsed_time;
}
