
#include "test_flann.h"

#include <flann/flann.h>

#include <iostream>
using namespace std;

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
	float* dataset;
	float* testset;
	int nn = 3;
	int* result;
	float* dists;
	struct FLANNParameters p;
	float speedup;
	flann_index_t index_id;

    int rows = (int)trainPoints.size();
    int cols = 1;
    int tcount = (int)testPoints.size();

    dataset = (float*)&trainPoints[0];
    testset = (float*)&testPoints[0];

    result = (int*) malloc(tcount*nn*sizeof(int));
    dists = (float*) malloc(tcount*nn*sizeof(float));

    p = DEFAULT_FLANN_PARAMETERS;
    p.algorithm = FLANN_INDEX_KDTREE;
    p.trees = 8;
    p.log_level = FLANN_LOG_INFO;
	p.checks = 64;

    index_id = flann_build_index(dataset, rows, cols, &speedup, &p);
    flann_find_nearest_neighbors_index(index_id, testset, tcount, result, dists, nn, &p);

    flann_free_index(index_id, &p);

    free(result);
    free(dists);
}
