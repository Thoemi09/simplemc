#include <fmt/base.h>
#include <simplemc/accs.hpp>

#include <random>

int main() {
    // AR(1) parameters and white noise distribution: phi = 0.9, sigma = 1.0
    const double phi = 0.9;
    const double sigma = 1.0;
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist(0.0, sigma);

    // accumulate samples into a batch accumulator
    simplemc::batch_acc<double> acc {};
    double x_t = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        acc << x_t;
        x_t = phi * x_t + normal_dist(rng);
    }

    // print the mean and variance of the accumulated data
    fmt::println("Mean: {}", acc.mean());
    fmt::println("Variance: {}", acc.variance());
}
