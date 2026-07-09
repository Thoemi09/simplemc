#include <fmt/ranges.h>
#include <simplemc/random.hpp>

#include <cmath>
#include <random>
#include <vector>

// Histogram on the interval [a, b] with uniform bin sizes.
struct histogram {
    // Constructor that takes the interval [a, b] and the number of bins.
    histogram(double a, double b, int nbins) : a(a), b(b), nbins(nbins), step((b - a) / nbins), data(nbins, 0.0) {}

    // Add a value to the histogram.
    void add(double value) {
        nsamples += 1;
        if (value < a || value >= b) {
            return;
        }
        int idx = static_cast<int>((value - a) / step);
        data[idx] += 1.0;
    }

    // Print the histogram + a reference function.
    void print(auto f) const {
        const auto norm = static_cast<double>(nsamples) * step;
        fmt::println("{:<15}{:<15}{:<15}", "y_i", "h(i)", "f(y_i)");
        for (int i = 0; i < nbins; ++i) {
            const auto x = a + step * (i + 0.5);
            fmt::println("{:<15.8f}{:<15.8f}{:<15.8f}", x, data[i] / norm, f(x));
        }
    }

    double a;
    double b;
    int nbins;
    double step;
    std::vector<double> data;
    long nsamples { 0 };
};

int main() {
    // construct a random number generator and a uniform distribution on [0, 1)
    simplemc::xoshiro256pp rng {};
    std::uniform_real_distribution uni01 { 0.0, 1.0 };

    // number of random samples to generate for each distribution and number of bins for the histograms
    const int nsamples = 1000000;
    int nbins = 10;

    // sample uniform distribution on [2, 5) and print histogram
    double a = 2;
    double b = 5;
    histogram hist_uni25 { a, b, nbins };
    for (int i = 0; i < nsamples; ++i) {
        hist_uni25.add(simplemc::uniform_sample(uni01(rng), a, b));
    }
    hist_uni25.print([a, b](auto) { return simplemc::uniform_pdf(a, b); });

    // sample exponential distribution on [2, 5) with lambda = 2.5 and print histogram
    double lambda = 2.5;
    histogram hist_exp { a, b, nbins };
    for (int i = 0; i < nsamples; ++i) {
        hist_exp.add(simplemc::exponential_sample(uni01(rng), lambda, a, b));
    }
    hist_exp.print([a, b, lambda](auto x) { return simplemc::exponential_pdf(x, lambda, a, b); });

    // sample safe exponential distribution on [2, 5) with lambda = -2.5 and print histogram
    lambda = -2.5;
    histogram hist_safe_exp { a, b, nbins };
    for (int i = 0; i < nsamples; ++i) {
        hist_safe_exp.add(simplemc::safe_exponential_sample(uni01(rng), lambda, a, b));
    }
    hist_safe_exp.print([a, b, lambda](auto x) { return simplemc::safe_exponential_pdf(x, lambda, a, b); });

    // sample normal distribution with mu = 0 and sigma = 1.5 and print histogram on [-5, 5]
    double mu = 0;
    double sigma = 1.5;
    a = -5;
    b = 5;
    histogram hist_normal { a, b, nbins };
    for (int i = 0; i < nsamples; ++i) {
        hist_normal.add(simplemc::normal_sample(rng, mu, sigma));
    }
    hist_normal.print([mu, sigma](auto x) { return simplemc::normal_pdf(x, mu, sigma); });

    // sample Cauchy distribution with x0 = -2 and gamma = 2.5 and print histogram on [-5, 1]
    double x0 = -2;
    double gamma = 2.5;
    a = -5;
    b = 1;
    histogram hist_cauchy { a, b, nbins };
    for (int i = 0; i < nsamples; ++i) {
        hist_cauchy.add(simplemc::cauchy_sample(uni01(rng), x0, gamma));
    }
    hist_cauchy.print([x0, gamma](auto x) { return simplemc::cauchy_pdf(x, x0, gamma); });

    // sample exclusive uniform integer distribution on {2, ..., 7} excluding y = 5 and print histogram
    int a_i = 2;
    int b_i = 7;
    int y = 5;
    nbins = 6;
    histogram hist_ex { a_i - 0.5, b_i + 0.5, nbins };
    for (int i = 0; i < nsamples; ++i) {
        hist_ex.add(simplemc::exclusive_uniform_int_sample(rng, a_i, b_i, y));
    }
    hist_ex.print([a_i, b_i, y](auto x) {
        return (std::abs(x - y) < 0.1 ? 0.0 : simplemc::exclusive_uniform_int_pdf(a_i, b_i));
    });
}
