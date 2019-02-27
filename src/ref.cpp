
#include "ref.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
float L2Norm(
    const float* const trainPoint,
    const float* const testPoint,
    const int& DIM)
{
    float sum = 0.0f;
    for (int d=0; d<DIM; ++d)
    {
        float diff = trainPoint[d] - testPoint[d];
        sum += diff * diff;
    }

    return sqrtf(sum);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Ref_kNN(
    const vector<float>& trainPoints,
    const vector<float>& testPoints,
    const int& DIM,
    const int& K,
    vector<float>& gt_distances,
    vector<int>&  gt_indices)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int size = (int)testPoints.size() / DIM;

    // Process one testPoints point at the time
    for (int tstId = 0; tstId < size; tstId++)
    {
        // local array to store all the distances / indexes for a given test point
        vector<float> distances(1, FLT_MAX);
        vector<int>   indices(1, 0);

        // Compute all distances / indexes
        for (int trnId = 0; trnId < trainSize; trnId++)
        {
            float distance = L2Norm(&trainPoints[trnId * DIM],
                                    &testPoints[tstId * DIM],
                                    DIM);

            for (int i = 0; i < distances.size(); i++)
            {
                // insertion sort
                if (distance < distances[i])
                {
                    distances.insert(distances.begin() + i, distance);
                    indices.insert(indices.begin() + i, trnId);

                    break;
                }
            }
        }

        // Copy K smallest distances and their associated index
        for (int k = 0; k < K; k++)
        {
            gt_distances[tstId * K + k]  = distances[k];
            gt_indices[tstId * K + k] = indices[k];
        }
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int CountMatches(
    const vector<float>& gt_distances,
    const vector<float>& test_distances,
    const int& K)
{
    int nb_correct = 0;

    int size = (int)gt_distances.size();

    for (int i = 0; i < size; i++)
    {
        float error = fabs(gt_distances[i] - test_distances[i]);

        if (error <= PRECISION)
            nb_correct++;
    }

    return nb_correct;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int CountMatches(
    const vector<int>& gt_indices,
    const vector<int>& test_indices,
    const int& K)
{
    int nb_correct = 0;

    int size = (int)gt_indices.size();

    for (int i = 0; i < size; i++)
    {
        int error = gt_indices[i] - test_indices[i];

        if (error == 0)
            nb_correct++;
    }

    return nb_correct;
}
