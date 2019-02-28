
#include "test_faiss.h"

#include "performance.h"
#include "ref.h"
#include "report.h"

#include <iomanip>
#include <iostream>
using namespace std;

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
