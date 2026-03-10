#include <fmt/ranges.h>
#include <simplemc/accs.hpp>

#include <random>

int main() {
    // normal distribution to be sampled: mu = 5, sigma = 1
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist(5.0, 1.0);

    // accumulate samples into a variance accumulator
    simplemc::var_acc<double> acc;
    for (int i = 0; i < 100000; ++i) {
        acc << normal_dist(rng);
    }

    // print the mean and variance of the accumulated data
    fmt::println("Mean: {}", acc.mean());
    fmt::println("Variance: {}", acc.variance_of_data());
}
