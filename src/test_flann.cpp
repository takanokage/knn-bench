
#include "test_flann.h"

#include "ref.h"
#include "report.h"

#include <flann/flann.h>
using namespace flann;

#include <sys/time.h>

#include <iomanip>
#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
double test_flann(
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

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances(testSize * K);
    vector<int> test_indices(testSize * K);

    Matrix<float> trainSet((float*)trainPoints.data(), trainSize, DIM);

    KDTreeSingleIndexParams params(15);
    Index<L2<float>> flann_index(trainSet, params);
    flann_index.buildIndex();

    Matrix<float> flann_query((float*)testPoints.data(), testSize, DIM);
    Matrix<int> flann_indices(test_indices.data(), flann_query.rows, K);
    Matrix<float> flann_distances(test_distances.data(), flann_query.rows, K);

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    for (int i = 0; i < nb_iterations; i++)
        int k = flann_index.knnSearch(flann_query, flann_indices, flann_distances,
                                    K, SearchParams(-1, 0.0));

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1e6;

    // Compute accuracy
    float distance_acc = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow(name,
        elapsed_time,
        nb_iterations,
        distance_acc,
        index_accuracy,
        validation);

    return elapsed_time;
}
