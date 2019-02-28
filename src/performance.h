#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <ctime>
#include <chrono>
#include <vector>

class Performance
{
private:
    // remove
    Performance() = delete;
    ~Performance() = delete;
    Performance(const Performance& gen) = delete;
    Performance(const Performance *const gen) = delete;
    Performance(Performance *const gen) = delete;
    Performance& operator=(const Performance& gen) = delete;
    Performance& operator=(const Performance *const gen) = delete;
    Performance& operator=(Performance *const gen) = delete;

private:
    // Store clock value on start.
    static std::vector<std::chrono::high_resolution_clock::time_point> start;
    // Store clock value on stop.
    static std::vector<std::chrono::high_resolution_clock::time_point> stop;

private:
    // Get the current counter.
    static std::chrono::high_resolution_clock::time_point Clock();

public:
    // Start ticking.
    static void Start();
    // Stop ticking.
    static void Stop();

    // Get the duration between Start & Stop in milliseconds.
    static double Duration();
};

#endif
