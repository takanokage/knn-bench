
#include "test_flann.h"
#include "report.h"

#include <flann/flann.h>

#include <sys/time.h>

#include <iomanip>
#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void test_flann(
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

	int nn = K;
	struct FLANNParameters p;
	float speedup;
	flann_index_t index_id;

    int rows = trainSize;
    int cols = 1;
    int tcount = testSize;

	float* dataset = (float*)&trainPoints[0];
	float* testset = (float*)&testPoints[0];

	int* result = (int*)&test_indices[0];
	float* dists = (float*)&test_distances[0];

    p = DEFAULT_FLANN_PARAMETERS;
    // p.algorithm = FLANN_INDEX_KDTREE;
    // p.trees = 8;
    // p.log_level = FLANN_LOG_INFO;
	// p.checks = 64;

    index_id = flann_build_index(dataset, rows, cols, &speedup, &p);

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    for (int i = 0; i < 1; i++)
        flann_find_nearest_neighbors_index(index_id, testset, tcount, result, dists, nn, &p);

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1e6;

    flann_free_index(index_id, &p);

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Verify both precisions and indexes of the K-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i = 0; i < testSize * K; i++)
    {
        if (fabs(test_distances[i] - gt_distances[i]) <= precision)
            nb_correct_precisions++;

        if (test_indices[i] == gt_indices[i])
            nb_correct_indexes++;
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) testSize * K);
    float index_accuracy     = nb_correct_indexes    / ((float) testSize * K);

    DisplayRow(name,
        elapsed_time,
        nb_iterations,
        precision_accuracy,
        index_accuracy);
}
