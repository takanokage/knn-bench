#include "performance.h"

using namespace std;
using namespace chrono;

using timestamp = high_resolution_clock::time_point;

// ----------------------------------------------------------------------------
// Store clock value on start.
// ----------------------------------------------------------------------------
vector<timestamp> Performance::start  = vector<timestamp>();

// ----------------------------------------------------------------------------
// Store clock value on stop.
// ----------------------------------------------------------------------------
vector<timestamp> Performance::stop   = vector<timestamp>();

// ----------------------------------------------------------------------------
// Get the current counter.
// ----------------------------------------------------------------------------
timestamp Performance::Clock()
{
    return high_resolution_clock::now();
}

// ----------------------------------------------------------------------------
// Start ticking.
// ----------------------------------------------------------------------------
void Performance::Start()
{
    start.push_back(Clock());
    stop.push_back(start.back());
}

// ----------------------------------------------------------------------------
// Stop ticking.
// ----------------------------------------------------------------------------
void Performance::Stop()
{
    stop.back() = Clock();
}

// ----------------------------------------------------------------------------
// Get the duration between Start & Stop in milliseconds.
// ----------------------------------------------------------------------------
double Performance::Duration()
{
    double us = duration_cast<microseconds>(stop.back() - start.back()).count();
    double ms = us / 1000.0;

    start.pop_back();
    stop.pop_back();

    return ms;
}
