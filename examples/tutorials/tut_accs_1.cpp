// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/accs.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

int main() {
    // Part 1: i.i.d. samples from a normal distribution N(mu_x, sigma_x^2)
    const double mu_x = 5.0;
    const double sigma_x = 1.0;
    const long n_x = 100000;
    const auto n_xd = static_cast<double>(n_x);
    std::mt19937_64 rng;
    std::normal_distribution<double> iid_dist(mu_x, sigma_x);

    // draw samples from the distribution
    std::vector<double> samples_x(n_x);
    for (long i = 0; i < n_x; ++i) {
        samples_x[i] = iid_dist(rng);
    }

    // manual two-pass reference: first compute the sample mean
    double sum = 0.0;
    for (double x : samples_x) {
        sum += x;
    }
    const double x_bar_ref = sum / n_xd;

    // then compute the standard error of the mean
    double sum_sq = 0.0;
    for (double x : samples_x) {
        const double d = x - x_bar_ref;
        sum_sq += d * d;
    }
    const double s_x_bar_ref = std::sqrt(sum_sq / (n_xd - 1) / n_xd);

    // accumulate the same data with a mean accumulator
    simplemc::mean_acc<double> macc_x;
    for (double x : samples_x) {
        macc_x << x;
    }

    // accumulate the same data with a variance accumulator
    simplemc::var_acc<double> vacc_x;
    for (double x : samples_x) {
        vacc_x << x;
    }

    // print results side by side: mean and standard error of the mean
    fmt::println("analytic:   mu_x  = {:<8.1f}, sigma_x_bar = {:.6f}", mu_x, sigma_x / std::sqrt(n_xd));
    fmt::println("reference:  x_bar = {:<8.6f}, s_x_bar     = {:.6f}", x_bar_ref, s_x_bar_ref);
    fmt::println("mean_acc:   x_bar = {:<8.6f}, s_x_bar     = N/A", macc_x.mean());
    fmt::println("var_acc:    x_bar = {:<8.6f}, s_x_bar     = {:.6f}", vacc_x.mean(), simplemc::stderror(vacc_x));
    fmt::println("");

    // parameters of the AR(1) process
    const double mu_y = 0.0;
    const double phi = 0.9;
    const double sigma_eps = 1.0;
    const long n_y = 1000000;
    const auto n_yd = static_cast<double>(n_y);
    std::normal_distribution<double> noise(mu_y, sigma_eps);

    // analytic reference results for the AR(1) process
    const double sigma2_y = sigma_eps * sigma_eps / (1.0 - phi * phi);
    const double tau_int = (1.0 + phi) / (1.0 - phi);
    const double sigma_y_bar = std::sqrt(sigma2_y * tau_int / n_yd);
    fmt::println("AR(1) analytic:");
    fmt::println("  y_bar       = {:.1f}", mu_y);
    fmt::println("  tau_int     = {:.1f}", tau_int);
    fmt::println("  sigma_y_bar = {:.2f}", sigma_y_bar);
    fmt::println("");

    // generate samples from the AR(1) process
    std::vector<double> samples_y(n_y);
    double y_t = 0.0;
    for (long i = 0; i < n_y; ++i) {
        y_t = phi * y_t + noise(rng);
        samples_y[i] = y_t;
    }

    // 1) naive var_acc with correlated samples
    simplemc::var_acc<double> vacc_y;
    for (double y : samples_y) {
        vacc_y << y;
    }

    fmt::println("AR(1) var_acc:");
    fmt::println("  y_bar   = {:.6f}", vacc_y.mean());
    fmt::println("  s_y_bar = {:.6f}", simplemc::stderror(vacc_y));
    fmt::println("");

    // 2) block_acc with a fixed block size B = 100
    simplemc::block_acc<simplemc::var_acc<double>> blkacc_y { 100 };
    for (double y : samples_y) {
        blkacc_y << y;
    }

    fmt::println("AR(1) block_acc:");
    fmt::println("  y_bar   = {:.6f}", blkacc_y.mean());
    fmt::println("  s_y_bar = {:.6f}", simplemc::stderror(blkacc_y));
    fmt::println("");

    // 3) autocorr_acc to scan a geometric hierarchy of block sizes
    simplemc::autocorr_acc<simplemc::var_acc<double>> aacc_y;
    for (double y : samples_y) {
        aacc_y << y;
    }

    // get the naive variance estimate and the highest level with at least 256 effective samples
    const auto s0 = aacc_y.accumulators()[0].variance_of_data();
    const auto max_level = aacc_y.find_level(256);

    // loop over levels and print the block size, effective sample size, standard error of the mean,
    // and tau estimate
    fmt::println("AR(1) autocorr_acc:");
    fmt::println("{:<8}{:<14}{:<14}{:<16}{:<14}", "l", "B_l", "N_eff", "s_y_bar", "tau");
    for (std::size_t l = 0; l <= max_level; ++l) {
        const auto& va = aacc_y.accumulators()[l];
        fmt::println("{:<8}{:<14}{:<14}{:<16.8f}{:<14.4f}", l, aacc_y.block_size(l), va.count(), simplemc::stderror(va),
            simplemc::accs::tau(s0, va.variance_of_data(), aacc_y.block_size(l)));
    }
    fmt::println("");

    // 4) batch_acc to group samples into a fixed number of batches (default: min 256, max 512)
    simplemc::batch_acc<double> bacc_y {};
    for (double y : samples_y) {
        bacc_y << y;
    }

    fmt::println("AR(1) batch_acc:");
    fmt::println("  y_bar   = {:.6f}", bacc_y.mean());
    fmt::println("  s_y_bar = {:.6f}", simplemc::stderror(bacc_y));
    fmt::println("  samples/batch = {}, N_eff = {}", bacc_y.batch_count(), bacc_y.num_full_batches());
}
