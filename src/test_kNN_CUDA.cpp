
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
bool test_kNN_CUDA(
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
    {
        bool passed = knn(
            &trainPoints[0],
            trainSize,
            &testPoints[0],
            testSize,
            DIM,
            K,
            &test_distances[0],
            &test_indices[0]);

        if (!passed)
            return false;
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1e6;

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Verify both precisions and indexes of the K-NN values
    int nb_correct_distances = CountMatches(gt_distances, test_distances, K);
    int nb_correct_indices   = CountMatches(gt_indices, test_indices, K);
    // for (int i = 0; i < testSize * K; i++)
    // {
    //     if (fabs(test_distances[i] - gt_distances[i]) <= precision)
    //         nb_correct_distances++;

    //     if (test_indices[i] == gt_indices[i])
    //         nb_correct_indices++;
    // }

    // Compute accuracy
    float precision_accuracy = nb_correct_distances / ((float) testSize * K);
    float index_accuracy     = nb_correct_indices   / ((float) testSize * K);

    DisplayRow(name,
        elapsed_time,
        nb_iterations,
        precision_accuracy,
        index_accuracy);

    return true;
}
