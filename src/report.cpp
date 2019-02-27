
#include "report.h"

#include<iomanip>
#include<iostream>
#include<string>
using namespace std;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void DisplayHeader()
{
    cout << setw(REPORT_WIDTH) << "Implementation";
    cout << setw(REPORT_WIDTH) << "Duration (s)";
    cout << setw(REPORT_WIDTH) << "Nr. iterations";
    cout << setw(REPORT_WIDTH) << "Precision accuracy";
    cout << setw(REPORT_WIDTH) << "Index accuracy";
    cout << setw(REPORT_WIDTH) << "Validation";
    cout << endl;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void DisplayRow(
    const char* const name,
    const double& elapsed_time,
    const int& nb_iterations,
    const double& precision_accuracy,
    const double& index_accuracy)
{
    // percentage of correct values required
    const float min_accuracy = 0.999f;

    cout << setw(REPORT_WIDTH) << name;
    cout << setw(REPORT_WIDTH) << right << setprecision(REPORT_PRECISION)
         << elapsed_time / nb_iterations;
    cout << setw(REPORT_WIDTH) << right << nb_iterations;
    cout << setw(REPORT_WIDTH) << right << precision_accuracy;
    cout << setw(REPORT_WIDTH) << right << index_accuracy;
    if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy )
        cout << setw(REPORT_WIDTH) << "PASSED";
    else
        cout << setw(REPORT_WIDTH) << "FAILED";

    cout << endl;

}
