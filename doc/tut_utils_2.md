@page tut_utils_2 Utilities Tutorial 2: Multi-dimensional indexing

[TOC]

In this tutorial, we show how to convert multi-dimensional indices to flat indices using the
simplemc::flat_index function from the @ref simplemc-utils library.

We further make use of the simplemc::row_major and simplemc::column_major tags to apply a certain
memory layout.

```cpp
#include <simplemc/utils.hpp>

#include <fmt/core.h>

#include <array>
#include <numeric>
#include <vector>

// Print a matrix with a given shape and memory order.
void print_matrix(const std::vector<int>& data, const std::array<int, 2>& shape, simplemc::nd_order auto order) {
    std::array<int, 2> idxs {};
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            idxs = { i, j };
            fmt::print("{:3} ", data[simplemc::flat_index(idxs, shape, order)]);
        }
        fmt::print("\n");
    }
    fmt::print("\n");
}

int main() {
    // create an integer vector and fill it with 1, ..., 15
    std::vector<int> data(15);
    std::iota(data.begin(), data.end(), 1);

    // print the data as a 5x3 matrix in row- and column-major order
    std::array<int, 2> shape {5, 3};
    fmt::print("5x3 row-major order:\n");
    print_matrix(data, shape, simplemc::row_major {});

    fmt::print("3x5 column-major order:\n");
    print_matrix(data, shape, simplemc::column_major {});
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

5x3 column-major order:
  1   6  11
  2   7  12
  3   8  13
  4   9  14
  5  10  15
```
