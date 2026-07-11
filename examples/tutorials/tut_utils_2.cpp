// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/utils.hpp>

#include <array>
#include <numeric>
#include <vector>

int main() {
    // create an integer vector and fill it with 1, ..., 15
    std::vector<int> data(15);
    std::iota(data.begin(), data.end(), 1);

    // print the vector to stdout
    fmt::println("{}", data);

    // generic lambda to print a matrix with a given shape and memory layout
    auto print_matrix = [](const auto& data, const auto& shape, auto mem_layout) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                std::array<int, 2> idxs = { i, j };
                fmt::print("{:3} ", data[simplemc::flat_index(idxs, shape, mem_layout)]);
            }
            fmt::println("");
        }
        fmt::println("");
    };

    // shape of the 5x3 matrix
    std::array<int, 2> shape1 { 5, 3 };

    // print the data as a 5x3 matrix in row-major order
    fmt::println("5x3 row-major order:");
    print_matrix(data, shape1, simplemc::row_major {});

    // print the data as a 5x3 matrix in column-major order
    fmt::println("5x3 column-major order:");
    print_matrix(data, shape1, simplemc::column_major {});

    // shape of the 3x5 matrix
    std::array<int, 2> shape2 { 3, 5 };

    // print the data as a 3x5 matrix in row-major order
    fmt::println("3x5 row-major order:");
    print_matrix(data, shape2, simplemc::row_major {});

    // print the data as a 3x5 matrix in column-major order
    fmt::println("3x5 column-major order:");
    print_matrix(data, shape2, simplemc::column_major {});
}
