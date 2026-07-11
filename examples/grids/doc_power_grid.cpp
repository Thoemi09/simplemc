// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/grids.hpp>

int main() {
    // create a quadratic grid of size 5 on [0, 1]
    simplemc::power_grid grid { 0.0, 1.0, 5, 2 };

    // print the grid points, bin centers and bin volumes (sizes)
    fmt::println("Grid points: {}", grid);
    fmt::println("Bin centers: {}", simplemc::bin_center_view(grid));
    fmt::println("Bin volumes: {}", simplemc::bin_volume_view(grid));
}
