// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/random.hpp>

int main() {
    // initialize xoshiro256ss RNG
    simplemc::xoshiro256ss rng {};

    // print 5 random numbers
    fmt::println("Random numbers in the interval [{}, {}]:", rng.min(), rng.max());
    for (int i = 0; i < 5; ++i) {
        fmt::println("{}", rng());
    }
}
