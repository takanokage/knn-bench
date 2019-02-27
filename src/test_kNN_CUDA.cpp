
#include "test_kNN_CUDA.h"

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

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances (testSize * K);
    vector<int>   test_indices(testSize * K);

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    // Compute K-NN several times
    for (int i=0; i<nb_iterations; ++i)
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
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    // Verify both precisions and indexes of the K-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<testSize*K; ++i)
    {
        if (fabs(test_distances[i] - gt_distances[i]) <= precision)
            nb_correct_precisions++;

        if (test_indices[i] == gt_indices[i])
            nb_correct_indexes++;
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) testSize * K);
    float index_accuracy     = nb_correct_indexes    / ((float) testSize * K);

    // Display report
    int width = 16;
    cout << setw(width) << name;
    cout << setw(width) << right << setprecision(5) << elapsed_time / nb_iterations;
    cout << setw(width) << right << nb_iterations;
    if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy )
        cout << setw(width) << "PASSED";
    else
        cout << setw(width) << "FAILED";
    cout << endl;

    return true;
}
