// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric/linear_interpolation.hpp>
#include <simplemc/utils/ranges.hpp>

#include <vector>

int main() {
    // define the function f(x) = x^2 that we want to interpolate
    auto f = [](double x) { return x * x; };

    // create a linear interpolation grid of size 50 on [0, 2]
    simplemc::linear_grid interp_grid { 0, 2, 50 };

    // obtain the function values at the grid points
    auto fview = simplemc::ranges::transform_view(interp_grid, [&f](double x) { return f(x); });
    auto fvals = std::vector<double>(fview.begin(), fview.end());

    // create the interpolation object for f(x) on the interpolation grid
    auto interp = simplemc::linear_interpolation { interp_grid, fvals };

    // create the grid at which we want to test our interpolation
    simplemc::linear_grid test_grid { 0, 2, 11 };

    // test the interpolation of the function f(x)
    fmt::println("{:<10}{:<20}{:<20}", "x", "interp(x)", "f(x)");
    for (auto x : test_grid) {
        fmt::println("{:<10.1f}{:<20.8g}{:<20.8g}", x, interp(x), f(x));
    }
    fmt::println("");
}
