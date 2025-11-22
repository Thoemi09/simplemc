#include <fmt/ranges.h>
#include <simplemc/grids.hpp>

int main() {
    // create a symmetric quadratic grid of size 9 on [0, 1]
    simplemc::symmetric_power_grid grid { 0.0, 1.0, 9, 2 };

    // print the grid points, bin centers and bin volumes (sizes)
    fmt::println("Grid points: {}", grid);
    fmt::println("Bin centers: {}", simplemc::bin_center_view(grid));
    fmt::println("Bin volumes: {}", simplemc::bin_volume_view(grid));
}
