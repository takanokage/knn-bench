
#pragma once

#include <cstdlib>

#define REPORT_PRECISION 5
#define REPORT_WIDTH     16

void DisplayHeader(const bool& validation = false);

void DisplayRow(
    const char* const name,
    const double& elapsed_time,
    const int& nb_iterations,
    const double& distance_acc,
    const double& index_accuracy,
    const bool& validation = true);
