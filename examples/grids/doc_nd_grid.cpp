// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/grids.hpp>

int main() {
    // create a linear grid of size 3 on [0, 1]
    simplemc::linear_grid lg { 0.0, 1.0, 3 };

    // create a quadratic grid of size 3 on [0, 0.5]
    simplemc::power_grid pg { 0.0, 0.5, 3, 2 };

    // combine the two 1d grids into a 2d grid
    simplemc::nd_grid grid_2d { lg, pg };

    // print some basic information
    fmt::println("Size: {}, Shape: {}", grid_2d.size(), grid_2d.shape());

    // print the grid points, bin centers and bin volumes (sizes)
    fmt::println("Grid points: {}", grid_2d);
    fmt::println("Bin centers: {}", simplemc::bin_center_view(grid_2d));
    fmt::println("Bin volumes: {}", simplemc::bin_volume_view(grid_2d));
}
