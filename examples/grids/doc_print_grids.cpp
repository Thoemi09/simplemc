#include <fmt/ranges.h>
#include <simplemc/grids.hpp>

int main() {
    // create a 1-dimensional grid of size 3
    simplemc::linear_grid grid_1d { 0.0, 1.0, 3 };
    fmt::println("1d: {}", simplemc::grid_view(grid_1d));

    // create a 2-dimensional grid of size 3x3
    simplemc::nd_grid grid_2d { grid_1d, grid_1d };
    fmt::println("2d: {}", simplemc::grid_view(grid_2d));
}
