
#include "test_flann.h"

#include "ref.h"
#include "report.h"

#include <flann/flann.h>

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
    const int& nb_iterations)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    // Allocate memory for computed K-NN neighbors
    vector<float> test_distances(testSize * K);
    vector<int> test_indices(testSize * K);

	float* dataset = (float*)&trainPoints[0];
	float* testset = (float*)&testPoints[0];

	int* result = (int*)&test_indices[0];
	float* dists = (float*)&test_distances[0];

	struct FLANNParameters p = DEFAULT_FLANN_PARAMETERS;
    p.algorithm = FLANN_INDEX_KDTREE;
    p.trees = 16;
    p.log_level = FLANN_LOG_INFO;
	p.checks = 64;

	float speedup;
	flann_index_t index_id = flann_build_index(dataset, trainSize, DIM, &speedup, &p);

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    for (int i = 0; i < nb_iterations; i++)
        flann_find_nearest_neighbors_index(index_id, testset, testSize, result, dists, K, &p);

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1e6;

    flann_free_index(index_id, &p);

    // Compute accuracy
    float precision_accuracy = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy     = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow(name,
        elapsed_time,
        nb_iterations,
        precision_accuracy,
        index_accuracy);

    return elapsed_time;
}
