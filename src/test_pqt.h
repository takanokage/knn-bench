
#pragma once

#include <vector>

double test_pqt(
    const std::vector<float>& trainPoints,
    const std::vector<float>& testPoints,
    const int& DIM,
    const int& K,
    const std::vector<float>& gt_distances,
    const std::vector<int>& gt_indices,
    const int& nb_iterations,
    const bool& validation = true);
