
#include "test_flann.h"

#include <flann/flann.h>

#include <stdio.h>
#include <stdlib.h>

using namespace flann;

void test_flann()
{
	float* dataset;
	float* testset;
	int nn;
	int* result;
	float* dists;
	struct FLANNParameters p;
	float speedup;
	flann_index_t index_id;

    int rows = 9000;
    int cols = 128;
    int tcount = 1000;

    /*
     * The files dataset.dat and testset.dat can be downloaded from:
     * http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/dataset.dat
     * http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/testset.dat
     */
    // printf("Reading input data file.\n");
    // dataset = read_points("dataset.dat", rows, cols);
    // printf("Reading test data file.\n");
    // testset = read_points("testset.dat", tcount, cols);

    nn = 3;
    result = (int*) malloc(tcount*nn*sizeof(int));
    dists = (float*) malloc(tcount*nn*sizeof(float));

    p = DEFAULT_FLANN_PARAMETERS;
    p.algorithm = FLANN_INDEX_KDTREE;
    p.trees = 8;
    p.log_level = FLANN_LOG_INFO;
	p.checks = 64;

    printf("Computing index.\n");
    index_id = flann_build_index(dataset, rows, cols, &speedup, &p);
    flann_find_nearest_neighbors_index(index_id, testset, tcount, result, dists, nn, &p);

    // write_results("results.dat",result, tcount, nn);

    flann_free_index(index_id, &p);
    free(dataset);
    free(testset);
    free(result);
    free(dists);
}
