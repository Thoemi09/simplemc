#include <fmt/ranges.h>
#include <simplemc/grids.hpp>
#include <simplemc/utils/ranges.hpp>

#include <random>
#include <array>

// Print a view in matrix form with the given shape.
void print_2d_view(auto&& view, long rows, long cols) {
    using namespace simplemc::ranges::views;
    for (long i = 0; i < rows; ++i) {
        fmt::println("{:n}", view | drop(i * cols) | take(cols));
    }
    fmt::println("");
}

int main() {
    // 1D custom grid of size 3 with points 2, 4 and 8
    simplemc::custom_grid grid1_1d { { 2, 4, 8 } };

    // 1D custom grid of size 3 with points 1, 3 and 9
    simplemc::custom_grid grid2_1d { { 1, 3, 9 } };

    // construct the 2D grid
    simplemc::nd_grid grid_2d { grid1_1d, grid2_1d };

    // print basic information
    const auto shape = grid_2d.shape();
    fmt::println("{}D grid of size {} and shape {}\n", grid_2d.dim(), grid_2d.size(), shape);

    // print grid points
    fmt::println("Grid points:");
    print_2d_view(grid_2d, shape[0], shape[1]);

    // print bin centers
    fmt::println("Bin centers: {}\n", simplemc::bin_center_view(grid_2d));

    // print bin volumes
    fmt::println("Bin volumes: {}\n", simplemc::bin_volume_view(grid_2d));

    // find the bin b_i to which a value x belongs
    std::mt19937 rng {};
    std::uniform_real_distribution dist1 { grid_2d.first()[0], grid_2d.last()[0] };
    std::uniform_real_distribution dist2 { grid_2d.first()[1], grid_2d.last()[1] };
    fmt::println("Find the bin:");
    for (int i = 0; i < 10; ++i) {
        const auto x = std::array<double, 2> { dist1(rng), dist2(rng) };
        const auto idx = grid_2d.index(x);
        const auto gidx = grid_2d.at(idx);
        const auto gidxp1 = grid_2d.at(idx[0] + 1, idx[1] + 1);
        fmt::println(
            "x = {::.4f} lies in bin [{},{}) x [{},{}) with index {}", x, gidx[0], gidxp1[0], gidx[1], gidxp1[1], idx);
    }
}
