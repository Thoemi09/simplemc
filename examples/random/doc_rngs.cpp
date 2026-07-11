// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/random.hpp>

#include <random>

int main() {
    // construct a xoshiro RNG
    simplemc::xoshiro256pp rng {};

    // print 5 random samples generated from a normal distribution
    std::normal_distribution<> dist { 0.0, 1.0 };
    for (int i = 0; i < 5; ++i) {
        fmt::println("{}", dist(rng));
    }
}
