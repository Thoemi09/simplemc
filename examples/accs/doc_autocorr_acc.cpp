// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/accs.hpp>

#include <random>

int main() {
    // AR(1) parameters and white noise distribution: phi = 0.9, sigma = 1.0
    const double phi = 0.9;
    const double sigma = 1.0;
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist(0.0, sigma);

    // accumulate samples into an autocorrelation accumulator
    simplemc::autocorr_acc<simplemc::var_acc<double>> acc;
    double x_t = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        acc << x_t;
        x_t = phi * x_t + normal_dist(rng);
    }

    // inspect how the variance and tau increase with block size
    auto const s0 = acc.accumulators()[0].variance_of_data();
    auto const max_level = acc.find_level(256);
    fmt::println("{:<15}{:<15}{:<15}{:<15}", "Count", "Block size", "Variance", "tau");
    for (std::size_t i = 0; i <= max_level; ++i) {
        const auto& va = acc.accumulators()[i];
        fmt::println("{:<15}{:<15}{:<15.8f}{:<15.8f}", va.count(), acc.block_size(i), va.variance(),
            simplemc::accs::tau(s0, va.variance_of_data(), acc.block_size(i)));
    }
}
