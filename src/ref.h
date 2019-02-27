
#pragma once

#include "types.h"
#include <vector>

float L2Norm(
    const float* const trainPoint,
    const float* const testPoint,
    const int& DIM);

float L2Norm(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& trnId,
    const int& tstId);

void Ref_kNN(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& dim,
    const int& k,
    std::vector<float>& gt_distances,
    std::vector<int>& gt_indices);
