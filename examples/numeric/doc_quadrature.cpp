// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/numeric.hpp>

#include <cmath>
#include <numbers>

int main() {
    using std::numbers::pi;

    // function we want to integrate
    auto f = [](double x) { return std::sin(x); };

    // integrate sin(x) from 0 to pi using the trapezoidal and Simpson's rule
    auto [val_t, diff_t, n_t] = simplemc::trapez_quadrature(f, 0, pi);
    auto [val_s, diff_s, n_s] = simplemc::simpson_quadrature(f, 0, pi);

    // output results
    fmt::println("{:<20}{:<20}{:<20}{:<5}", "Rule", "I_N", "|I_N - I_{N-1}|", "N");
    fmt::println("{:<20}{:<20.8f}{:<20.8f}{:<5}", "Trapezoidal", val_t, diff_t, n_t);
    fmt::println("{:<20}{:<20.8f}{:<20.8f}{:<5}", "Simpson", val_s, diff_s, n_s);
}
