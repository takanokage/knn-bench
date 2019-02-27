
#pragma once

#include <cstdlib>

#define REPORT_PRECISION 5
#define REPORT_WIDTH     16

void DisplayHeader();

void DisplayRow(
    const char* const name,
    const double& elapsed_time,
    const int& nb_iterations,
    const double& precision_accuracy,
    const double& index_accuracy);
