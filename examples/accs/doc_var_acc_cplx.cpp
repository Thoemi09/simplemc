#include <fmt/base.h>
#include <fmt/std.h>
#include <simplemc/accs.hpp>

#include <complex>
#include <random>

int main() {
    // complex distribution to be sampled:
    // - real part: normally distributed with mu = 5 and sigma = 1
    // - imaginary part: normally distributed with mu = -3 and sigma = 0.5
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist_r(5.0, 1.0);
    std::normal_distribution<double> normal_dist_i(-3.0, 0.5);

    // accumulate samples into a variance accumulator
    simplemc::var_acc<std::complex<double>> acc;
    for (int i = 0; i < 100000; ++i) {
        acc << std::complex<double> { normal_dist_r(rng), normal_dist_i(rng) };
    }

    // print the mean and variance of the real/imaginary parts of the accumulated data
    fmt::println("Mean: {:.5f}", acc.mean());
    fmt::println("Variance of real part: {:.5f}", acc.variance_of_real_data());
    fmt::println("Variance of imag part: {:.5f}", acc.variance_of_imag_data());
}
