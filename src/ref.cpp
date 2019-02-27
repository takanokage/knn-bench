
#include "ref.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <iomanip>
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
    for (int d = 0; d < DIM; d++)
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

    int size = (int)gt_distances.size() / K;

    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < K; k++)
        {
            float error = fabs(gt_distances[i * K + k] - test_distances[i * K + k]);

            if (error <= PRECISION)
                nb_correct++;
        }
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

    int size = (int)gt_indices.size() / K;

    for (int i = 0; i < size; i++)
    {
        vector<int> lgt_indices(K);
        memcpy(&lgt_indices[0], &gt_indices[i * K], K * sizeof(float));
        sort(lgt_indices.begin(), lgt_indices.end());

        vector<int> ltest_indices(K);
        memcpy(&ltest_indices[0], &test_indices[i * K], K * sizeof(float));
        sort(ltest_indices.begin(), ltest_indices.end());

        vector<int> intersection(2 * K);
        vector<int>::iterator it;
        it = set_intersection(lgt_indices.begin(), lgt_indices.end(),
                              ltest_indices.begin(), ltest_indices.end(),
                              intersection.begin());
        intersection.resize(it - intersection.begin());
        nb_correct += intersection.size();
    }

    return nb_correct;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
float ComputeAccuracy(
    const vector<float>& gt_distances,
    const vector<float>& test_distances,
    const int& K)
{
    int nb_correct = CountMatches(gt_distances, test_distances, K);

    float accuracy = (float)nb_correct / gt_distances.size();

    return accuracy;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
float ComputeAccuracy(
    const vector<int>& gt_indices,
    const vector<int>& test_indices,
    const int& K)
{
    int nb_correct = CountMatches(gt_indices, test_indices, K);

    float accuracy = (float)nb_correct / gt_indices.size();

    return accuracy;
}
