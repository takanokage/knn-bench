
#pragma once

#include <vector>

void test_flann(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const std::vector<float>& gt_distances,
    const std::vector<int>& gt_indices,
    const char* const name,
    const int& nb_iterations);
