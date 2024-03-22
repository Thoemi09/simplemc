/**
 * @file
 * @brief Implementation of a telegraph process for testing the accumulators.
 */

#ifndef SIMPLEMC_TESTS_ACCS_TELEGRAPH_PROCESS_HPP
#define SIMPLEMC_TESTS_ACCS_TELEGRAPH_PROCESS_HPP

#include <simplemc/utils/concepts.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>

// Calculate the sample mean.
template <simplemc::double_or_complex T>
[[nodiscard]] auto sample_mean(const std::vector<T>& samples) {
    T sum = 0.0;
    for (auto s : samples) {
        sum += s;
    }
    return sum / static_cast<T>(samples.size());
}

// Calculate the sample variance.
[[nodiscard]] auto sample_variance(const std::vector<double>& samples) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (auto s : samples) {
        sum += s;
        sum_sq += s * s;
    }
    auto tc = static_cast<double>(samples.size());
    return (sum_sq - sum * sum / tc) / (tc - 1);
}

[[nodiscard]] auto sample_variance(const std::vector<std::complex<double>>& samples) {
    double re_sum = 0.0, im_sum = 0.0, re_im_sum = 0.0;
    double re_sum_sq = 0.0, im_sum_sq = 0.0;
    for (auto s : samples) {
        re_sum += std::real(s);
        re_sum_sq += std::real(s) * std::real(s);
        im_sum += std::imag(s);
        im_sum_sq += std::imag(s) * std::imag(s);
        re_im_sum += std::real(s) * std::imag(s);
    }
    auto tc = static_cast<double>(samples.size());
    const auto re_var = (re_sum_sq - re_sum * re_sum / tc) / (tc - 1);
    const auto im_var = (im_sum_sq - im_sum * im_sum / tc) / (tc - 1);
    const auto re_im_cov = (re_im_sum - re_sum * im_sum / tc) / (tc - 1);
    return std::make_tuple(re_var, im_var, re_im_cov);
}

// Implementation of a telegraph process, i.e. dichotomous Markov process.
// See https://en.wikipedia.org/wiki/Telegraph_process.
template <simplemc::double_or_complex T>
struct telegraph_process {
    // Value type of the states.
    using value_type = T;

    // The type of the distribution used to generate a new state.
    using dist_t = std::discrete_distribution<std::size_t>;

    // Set the transition probabilities.
    void set_probabilities(double p0, double p1) {
        probs[0] = p0;
        probs[1] = p1;
        dists[0] = dist_t { 1.0 - probs[0], probs[0] };
        dists[1] = dist_t { probs[1], 1.0 - probs[1] };
    }

    // Set the states.
    void set_states(value_type s0, value_type s1) {
        states[0] = s0;
        states[1] = s1;
    }

    // Perform a single Markov step.
    void do_step() { current = dists[current](eng); }

    // Warm up the Markov chain.
    void warm_up(int steps) {
        for (int i = 0; i < steps; ++i) {
            do_step();
        }
    }

    // Simulate the Markov chain for a given number of steps.
    void simulate(int steps) {
        samples.resize(steps);
        for (int i = 0; i < steps; ++i) {
            do_step();
            ++counts[current];
            ++total_count;
            samples[i] = current;
        }
    }

    // Get the sampled values.
    [[nodiscard]] auto get_sampled_values() const {
        std::vector<value_type> res;
        for (auto s : samples) {
            res.push_back(states[s]);
        }
        return res;
    }

    // Get analytic stationary distribution.
    [[nodiscard]] auto stationary_distribution() const {
        return std::array<double, 2> { probs[1] / (probs[0] + probs[1]), probs[0] / (probs[0] + probs[1]) };
    }

    // Get analytic mean.
    [[nodiscard]] auto mean() const { return (probs[1] * states[0] + probs[0] * states[1]) / (probs[0] + probs[1]); }

    // Get variance.
    [[nodiscard]] auto variance() const {
        const auto psum = probs[0] + probs[1];
        const auto sdiff = states[0] - states[1];
        if constexpr (std::is_same_v<value_type, double>) {
            return sdiff * sdiff * probs[0] * probs[1] / (psum * psum);
        } else {
            const auto re_sdiff = std::real(sdiff);
            const auto im_sdiff = std::imag(sdiff);
            const auto re_var = re_sdiff * re_sdiff * probs[0] * probs[1] / (psum * psum);
            const auto im_var = im_sdiff * im_sdiff * probs[0] * probs[1] / (psum * psum);
            const auto re_im_cov = re_sdiff * im_sdiff * probs[0] * probs[1] / (psum * psum);
            return std::make_tuple(re_var, im_var, re_im_cov);
        }
    }

    // [[nodiscard]] auto autocorr(std::uint64_t frac = 100) const {
    //     const auto [m, m_est] = mean();
    //     const auto [v, v_est] = variance();
    //     const auto corr_count = total_count / frac;
    //     std::vector<double> est_corr(corr_count, 0.0);
    //     double est_tau = 0.0;
    //     for (std::size_t i = 0; i < corr_count; ++i) {
    //         const auto id = static_cast<double>(i);
    //         for (std::size_t j = 0; j < total_count - i; ++j) {
    //             est_corr[i] += (samples[j] - m) * (samples[j + i] - m) / static_cast<double>(total_count - i) / v;
    //         }
    //         est_tau += (1.0 - id / static_cast<double>(total_count)) * est_corr[i];

    //     }
    //     return std::make_tuple(est_tau, est_corr);
    // }

    std::array<value_type, 2> states { -1.0, 1.0 };
    std::array<double, 2> probs { 0.5, 0.5 };
    std::array<std::uint64_t, 2> counts { 0, 0 };
    std::uint64_t total_count { 0 };
    std::size_t current { 0 };
    std::mt19937 eng { 0x8a34e234 };
    std::array<dist_t, 2> dists { dist_t { 1.0 - probs[0], probs[0] }, dist_t { probs[1], 1.0 - probs[1] } };
    std::vector<std::size_t> samples;
};

#endif // SIMPLEMC_TESTS_ACCS_TELEGRAPH_PROCESS_HPP