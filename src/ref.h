
#pragma once

#include "types.h"
#include <vector>

#define PRECISION        1e-3f
#define ACCURACY         0.999f

float L2Norm(
    const float* const trainPoint,
    const float* const testPoint,
    const int& DIM);

void Ref_kNN(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& dim,
    const int& k,
    std::vector<float>& gt_distances,
    std::vector<int>& gt_indices);

int CountMatches(
    const std::vector<float>& gt_distances,
    const std::vector<float>& test_distances,
    const int& K);

int CountMatches(
    const std::vector<int>& gt_indices,
    const std::vector<int>& test_indices,
    const int& K);
