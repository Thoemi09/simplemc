// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/grids.hpp>
#include <simplemc/utils/ranges.hpp>

#include <random>
#include <string_view>
#include <vector>

// Test a given grid.
void test_grid(const auto& grid, std::string_view name) {
    // print basic information
    fmt::println("{} grid of size {} on the interval [{},{}]\n", name, grid.size(), grid.first(), grid.last());

    // print the grid points, bin centers and bin volumes (sizes)
    fmt::println("Grid points: {::.4g}", grid);
    fmt::println("Bin centers: {::.4g}", simplemc::bin_center_view(grid));
    fmt::println("Bin volumes: {::.4g}\n", simplemc::bin_volume_view(grid));

    // find the bin b_i to which a value x belongs
    std::mt19937 rng {};
    std::uniform_real_distribution dist { grid.first(), grid.last() };
    fmt::println("Find the bin:");
    for (int i = 0; i < 10; ++i) {
        const auto x = dist(rng);
        const auto idx = grid.index(x);
        fmt::println("x = {:.4f} lies in bin [{:.4g},{:.4g}) with index {}", x, grid.at(idx), grid.at(idx + 1), idx);
    }
    fmt::println("");
}

int main() {
    // linear grid
    test_grid(simplemc::linear_grid { 0, 10, 11 }, "linear");

    // quadratic grid
    test_grid(simplemc::power_grid { 0, 10, 11, 2 }, "quadratic");

    // cubic grid
    test_grid(simplemc::power_grid { 0, 10, 11, 3 }, "cubic");

    // symmetric quadratic grid
    test_grid(simplemc::symmetric_power_grid { 0, 10, 11, 2 }, "symmetric quadratic");

    // custom grid
    auto grid_pts = std::vector<double> { 0, 0.2, 0.8, 3.2, 4.9, 4.99, 6.1, 8.3, 8.7, 9, 10 };
    test_grid(simplemc::custom_grid { grid_pts }, "custom");
}
