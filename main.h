
#pragma once

#include <vector>

template<typename T>
std::vector<T> Transpose(
    const std::vector<T>& data,
    const int& rows,
    const int& cols)
{
    std::vector<T> dataT(rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dataT[c * rows + r] = data[r * cols + c];

    return dataT;
}
