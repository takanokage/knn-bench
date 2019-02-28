
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
// ----------------------------------------------------------------------------
double test_faiss(
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


    int d = 64;                            // dimension
    int nb = 100000;                       // database size
    int nq = 10000;                        // nb of queries

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    faiss::gpu::StandardGpuResources res;

    // Using a flat index

    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);

    printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
    index_flat.add(nb, xb);  // add vectors to the index
    printf("ntotal = %ld\n", index_flat.ntotal);

    int k = 4;

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        index_flat.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    // Using an IVF index

    int nlist = 100;
    faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search

    assert(!index_ivf.is_trained);
    index_ivf.train(nb, xb);
    assert(index_ivf.is_trained);
    index_ivf.add(nb, xb);  // add vectors to the index

    printf("is_trained = %s\n", index_ivf.is_trained ? "true" : "false");
    printf("ntotal = %ld\n", index_ivf.ntotal);

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        index_ivf.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }


    delete [] xb;
    delete [] xq;




    Performance::Start();

    for (int i = 0; i < nb_iterations; i++)
    {

    }

    Performance::Stop();
    double elapsed_time = Performance::Duration() / nb_iterations;

    // Compute accuracy
    float distance_acc = ComputeAccuracy(gt_distances, test_distances, K);
    float index_accuracy = ComputeAccuracy(gt_indices, test_indices, K);

    DisplayRow("faiss",
        elapsed_time,
        nb_iterations,
        distance_acc,
        index_accuracy,
        validation);

    return elapsed_time;
}
