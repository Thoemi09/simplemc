#include <fmt/ranges.h>
#include <simplemc/accs.hpp>
#include <simplemc/utils/to_string.hpp>

#include <random>

int main() {
    // normal distribution to be sampled: mu = 2, sigma = 1
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist(2.0, 1.0);

    // accumulate samples into a covariance accumulator of size 2
    simplemc::covar_acc_dynamic<double> acc{2};
    for (int i = 0; i < 100000; ++i) {
        auto sample = normal_dist(rng);
        acc << Eigen::Vector2d{ sample, 2.0 * sample };
    }

    // print the mean and covariance matrix of the accumulated data
    fmt::println("Mean: {}", acc.mean());
    fmt::println("Covariance:\n{}", simplemc::to_string(acc.covariance_of_data()));
}
