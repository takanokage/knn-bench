
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
    const vector<float>& trainPoints,
    const vector<float>& testPoints,
    const int& DIM,
    const int& trnId,
    const int& tstId)
{
    int trainSize = (int)trainPoints.size() / DIM;
    int testSize = (int)testPoints.size() / DIM;

    float sum = 0.f;
    for (int d=0; d<DIM; ++d)
    {
        const float diff = trainPoints[trnId * DIM + d] - testPoints[tstId * DIM + d];
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
    int testSize = (int)testPoints.size() / DIM;

    // Process one testPoints point at the time
    for (int tstId = 0; tstId < testSize; tstId++)
    {
        // local array to store all the distances / indexes for a given test point
        vector<float> distances(1, FLT_MAX);
        vector<int>   indices(1, 0);

        // Compute all distances / indexes
        for (int trnId = 0; trnId < trainSize; trnId++)
        {
            float distance = L2Norm(trainPoints, testPoints, DIM, trnId, tstId);

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
