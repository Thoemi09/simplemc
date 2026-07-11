// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/grids.hpp>

int main() {
    // create a custom grid of size 5
    simplemc::custom_grid grid { std::vector<double> { 1.0, 2.3, 2.4, 5.7, 100.0 } };

    // print the grid points, bin centers and bin volumes (sizes)
    fmt::println("Grid points: {::.4g}", grid);
    fmt::println("Bin centers: {::.4g}", simplemc::bin_center_view(grid));
    fmt::println("Bin volumes: {::.4g}", simplemc::bin_volume_view(grid));
}
