
#pragma once

#include "knncuda.h"

#include <vector>

double test_kNN_CUDA(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const std::vector<float>& gt_distances,
    const std::vector<int>& gt_indices,
    bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
    const char* const name,
    const int& nb_iterations);
