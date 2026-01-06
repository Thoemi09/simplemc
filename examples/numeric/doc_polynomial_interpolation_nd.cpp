#include <fmt/ranges.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric/polynomial_interpolation.hpp>
#include <simplemc/utils/nd_indexing.hpp>
#include <simplemc/utils/ranges.hpp>

#include <vector>

int main() {
    // define the function f(x, y) = x^2 + y^3 that we want to interpolate
    auto f = [](double x, double y) { return x * x + y * y * y; };

    // create a linear 2D interpolation grid of size 50 x 50 on [0, 2] x [0, 2]
    simplemc::linear_grid grid_1d { 0, 2, 50 };
    simplemc::nd_grid interp_grid { grid_1d, grid_1d };

    // obtain the function values at the grid points (in row-major order)
    auto fview = simplemc::ranges::transform_view(interp_grid, [&f](const auto& x) { return f(x[0], x[1]); });
    auto fvals = std::vector<double>(fview.begin(), fview.end());

    // create the interpolation object for f(x, y) on the interpolation grid
    auto interp = simplemc::polynomial_interpolation_nd { interp_grid, fvals, 2, simplemc::row_major {} };

    // create the grid at which we want to test our interpolation
    grid_1d.reset(0, 2, 11);
    simplemc::nd_grid test_grid { grid_1d, grid_1d };

    // test the interpolation of the function f(x, y)
    fmt::println("{:<15}{:<20}{:<20}", "(x, y)", "interp(x, y)", "f(x, y)");
    for (auto [x, y] : test_grid) {
        fmt::print("{:<15}{:<20.8g}{:<20.8g}\n", fmt::format("({:.2g},{:.2g})", x, y), interp(x, y), f(x, y));
    }
}
