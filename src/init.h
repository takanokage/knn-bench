
#pragma once

#include <cstdlib>

// initialize an array of something.
template<typename T>
void init(
    T* const data,
    const size_t& size,
    const T& vmin = (T)0,
    const T& vmax = (T)1)
{
    T factor = (vmax - vmin) / RAND_MAX;

    for (size_t i = 0; i < size; i++)
        data[i] = rand() * factor;
}
