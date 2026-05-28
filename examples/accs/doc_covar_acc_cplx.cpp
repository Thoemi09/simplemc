#include <fmt/ranges.h>
#include <fmt/std.h>
#include <simplemc/accs.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/to_string.hpp>

#include <complex>
#include <random>

int main() {
    // complex distribution to be sampled:
    // - real part: normal distributed with mu = 5 and sigma = 1
    // - imaginary part: normal distributed with mu = -3 and sigma = 0.5
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist_r(5.0, 1.0);
    std::normal_distribution<double> normal_dist_i(-3.0, 0.5);

    // accumulate samples into a covariance accumulator of size 2
    simplemc::covar_acc_dynamic<std::complex<double>> acc{2};
    for (int i = 0; i < 100000; ++i) {
        auto sample = std::complex<double>{ normal_dist_r(rng), normal_dist_i(rng) };
        acc << Eigen::Vector2cd{ sample, 2.0 * sample };
    }

    // print the mean and covariance matrices of the real/imaginary parts of the accumulated data
    fmt::println("Mean: {::.5f}", simplemc::make_span(acc.mean()));
    fmt::println("Covariance of real part:\n{}", simplemc::to_string(acc.covariance_of_real_data()));
    fmt::println("Covariance of imaginary part:\n{}", simplemc::to_string(acc.covariance_of_imag_data()));
}
