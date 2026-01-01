#include <fmt/base.h>
#include <simplemc/random.hpp>

int main() {
    // initialize splitmix64 RNG
    simplemc::splitmix64 rng {};

    // print 5 random numbers
    fmt::println("Random numbers in the interval [{}, {}]:", rng.min(), rng.max());
    for (int i = 0; i < 5; ++i) {
        fmt::println("{}", rng());
    }
}
