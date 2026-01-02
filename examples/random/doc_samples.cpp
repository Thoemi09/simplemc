#include <fmt/base.h>
#include <simplemc/random.hpp>

#include <random>

int main() {
    // construct a xoshiro256p RNG and a uniform distribution on [0, 1)
    std::mt19937 rng {};
    std::uniform_real_distribution<double> dist { 0.0, 1.0 };

    // generate 5 random samples from an exponential distribution \lambda e^{-\lambda x}
    // and print the corresponding values of the pdf
    const double lambda = 2.5;
    fmt::println("{:^10}{:^10}", "x", "f(x)");
    for (int i = 0; i < 5; ++i) {
        const auto x = simplemc::exponential_sample(dist(rng), lambda);
        fmt::println("{:^10.4f}{:^10.4f}", x, simplemc::exponential_pdf(x, lambda));
    }
}
