@page tut_utils_2 Utilities Tutorial 2: Multi-dimensional indexing

[TOC]

In this tutorial, we show how to convert multi-dimensional indices to flat indices using the simplemc::flat_index function from the
@ref simplemc-utils library.

We further make use of the simplemc::row_major and simplemc::column_major tags to apply a certain memory layout.

```cpp
#include <simplemc/utils.hpp>
#include <array>
#include <numeric>
#include <vector>

int main() {
    // create an integer vector and fill it with 1, ..., 15
    std::vector<int> data(15);
    std::iota(data.begin(), data.end(), 1);

    // shape and multi-dimensional index arrays
    std::array<int, 2> shape {};
    std::array<int, 2> idxs {};

    // interpret the data as 5x3 matrix in row-major order
    shape = { 5, 3 };
    fmt::print("5x3 row-major order:\n");
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            idxs = { i, j };
            fmt::print("{:3} ", data[simplemc::flat_index(idxs, shape, simplemc::row_major {})]);
        }
        fmt::print("\n");
    }
    fmt::print("\n");

    // interpret the data as 3x5 matrix in column-major order
    shape = { 3, 5 };
    fmt::print("3x5 column-major order:\n");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            idxs = { i, j };
            fmt::print("{:3} ", data[simplemc::flat_index(idxs, shape, simplemc::column_major {})]);
        }
        fmt::print("\n");
    }
}
```

Output:

```
5x3 row-major order:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15

3x5 column-major order:
  1   4   7  10  13
  2   5   8  11  14
  3   6   9  12  15
```
