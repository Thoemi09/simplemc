#include "simplemc/random/xoshiro256.hpp"
#include <fmt/ranges.h>
#include <random>
#include <simplemc/random/splitmix64.hpp>

int main() {
    std::seed_seq seq;
    simplemc::xoshiro256pp rng { seq };
    fmt::print("{}\n", rng.internal_state());
    for (int i = 0; i < 100; i++) {
        fmt::print("{}\n", rng());
    }
}
