
#pragma once

#include "types.h"
#include <vector>

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
