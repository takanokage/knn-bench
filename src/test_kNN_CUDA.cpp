
#include "test_kNN_CUDA.h"

#include "ref.h"
#include "report.h"

#include <cmath>
#include <sys/time.h>

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
    bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
    const char* const name,
    const int& nb_iterations)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances(testSize * K);
    vector<int> test_indices(testSize * K);

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

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

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1e6;

    // Compute accuracy
    float distance_acc = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy     = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow(name,
        elapsed_time,
        nb_iterations,
        distance_acc,
        index_accuracy);

    return elapsed_time;
}
