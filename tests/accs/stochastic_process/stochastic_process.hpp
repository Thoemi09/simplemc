#ifndef SIMPLEMC_TESTS_ACCS_STOCHASTIC_PROCESS_HPP
#define SIMPLEMC_TESTS_ACCS_STOCHASTIC_PROCESS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>

#include <fmt/ranges.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

// Implementation of a simple discrete stochastic process.
template <simplemc::double_or_complex T, long N>
struct stochastic_process {
    // Size of a single state.
    static constexpr auto state_size = N;

    // Value type of the states.
    using value_type = T;

    // Type of a single state.
    using state_type = Eigen::Array<value_type, N, 1>;

    // Double vector type.
    using dbl_vec_type = Eigen::Array<double, N, 1>;

    // Vector type.
    using vec_type = Eigen::Array<value_type, N, 1>;

    // Double matrix type.
    using dbl_mat_type = Eigen::Array<double, N, N>;

    // Matrix type.
    using mat_type = Eigen::Array<value_type, N, N>;

    // The type of the distribution used to generate a new state.
    using dist_type = std::discrete_distribution<std::size_t>;

    // Default constructor.
    stochastic_process() = default;

    // Construct a stochastic process with the given state space and target weights/probabilities.
    stochastic_process(std::vector<state_type> s, const std::vector<double>& w) :
        states(std::move(s)),
        counts(states.size(), 0),
        dist(w.begin(), w.end()),
        probs(dist.probabilities()) {}

    // Set the target weights/probabilities.
    void set_weights(const std::vector<double>& w) {
        dist = dist_type(w.begin(), w.end());
        probs = dist.probabilities();
    }

    // Set the MCMC proposal probabilities.
    void set_mcmc_proposal(const Eigen::MatrixXd& p) {
        mcmc_dists.clear();
        for (int i = 0; i < p.cols(); ++i) {
            mcmc_dists.emplace_back(p.col(i).begin(), p.col(i).end());
        }
    }

    // Set the state space.
    void set_states(std::vector<state_type> s) {
        states = std::move(s);
        counts.assign(states.size(), 0);
    }

    // Run the stochastic process for a given number of steps (uses direct sampling).
    void run(int steps) {
        samples.resize(steps);
        for (int i = 0; i < steps; ++i) {
            current = dist(eng);
            ++counts[current];
            ++total_count;
            samples[i] = states[current];
        }
    }

    // Make a single MCMC step.
    void mcmc_step() {
        const auto new_state = mcmc_dists[current](eng);
        double ratio = (probs[new_state] * mcmc_dists[new_state].probabilities()[current]) /
            (probs[current] * mcmc_dists[current].probabilities()[new_state]);
        if (uni01(eng) < ratio) {
            current = new_state;
        }
    }

    // Warm up the Markov chain.
    void mcmc_warmup(int steps) {
        for (int i = 0; i < steps; ++i) {
            mcmc_step();
        }
    }

    // Run an MCMC for a given number of steps.
    void mcmc_run(int steps) {
        samples.resize(steps);
        for (int i = 0; i < steps; ++i) {
            mcmc_step();
            ++counts[current];
            ++total_count;
            samples[i] = states[current];
        }
    }

    std::vector<state_type> states {}; // State space.
    std::vector<std::uint64_t> counts {}; // How often did I visit each state?
    std::uint64_t total_count { 0 }; // Total number of samples.
    std::size_t current { 0 }; // Current state.
    std::mt19937 eng { 0x8a34e234 }; // Random number engine.
    dist_type dist {}; // Target distribution.
    std::vector<double> probs {}; // Probabilities of the target distribution.
    std::vector<state_type> samples {}; // Samples.
    std::vector<dist_type> mcmc_dists {}; // MCMC proposal distributions.
    std::uniform_real_distribution<double> uni01 { 0.0, 1.0 }; // Uniform distribution.
};

// Calculate the analytic mean of the given stochastic process.
template <typename S>
[[nodiscard]] auto analytic_mean(const S& sp) {
    auto mean = S::vec_type::Zero().eval();
    for (const auto& [s, p] : simplemc::ranges::views::zip(sp.states, sp.probs)) {
        mean += p * s;
    }
    return mean;
}

// Calculate the sample mean of the given stochastic process.
template <typename S>
[[nodiscard]] auto sample_mean(const S& sp) {
    auto sum = S::vec_type::Zero().eval();
    for (const auto& s : sp.samples) {
        sum += s;
    }
    return (sum / static_cast<double>(sp.total_count)).eval();
}

// Calculate the analytic variance of the given stochastic process.
template <typename S>
    requires std::is_same_v<typename S::value_type, double>
[[nodiscard]] auto analytic_variance(const S& sp) {
    auto var = S::vec_type::Zero().eval();
    const auto mean = analytic_mean(sp);
    for (const auto& [s, p] : simplemc::ranges::views::zip(sp.states, sp.probs)) {
        var += p * (s - mean).square();
    }
    return var;
}

template <typename S>
    requires std::is_same_v<typename S::value_type, std::complex<double>>
[[nodiscard]] auto analytic_variance(const S& sp) {
    auto re_var = S::dbl_vec_type::Zero().eval();
    auto im_var = S::dbl_vec_type::Zero().eval();
    auto re_im_cov = S::dbl_vec_type::Zero().eval();
    const auto mean = analytic_mean(sp);
    for (const auto& [s, p] : simplemc::ranges::views::zip(sp.states, sp.probs)) {
        auto tmp = s - mean;
        re_var += p * tmp.real().square();
        im_var += p * tmp.imag().square();
        re_im_cov += p * tmp.real() * tmp.imag();
    }
    return std::make_tuple(re_var, im_var, re_im_cov);
}

// Calculate the sample variance of the given stochastic process.
template <typename S>
    requires std::is_same_v<typename S::value_type, double>
[[nodiscard]] auto sample_variance(const S& sp) {
    auto sum = S::vec_type::Zero().eval();
    auto sum_sq = S::vec_type::Zero().eval();
    for (const auto& s : sp.samples) {
        sum += s;
        sum_sq += s * s;
    }
    const auto tc = static_cast<double>(sp.total_count);
    return ((sum_sq - sum * sum / tc) / (tc - 1)).eval();
}

template <typename S>
    requires std::is_same_v<typename S::value_type, std::complex<double>>
[[nodiscard]] auto sample_variance(const S& sp) {
    auto re_sum = S::dbl_vec_type::Zero().eval();
    auto im_sum = S::dbl_vec_type::Zero().eval();
    auto re_sum_sq = S::dbl_vec_type::Zero().eval();
    auto im_sum_sq = S::dbl_vec_type::Zero().eval();
    auto re_im_sum = S::dbl_vec_type::Zero().eval();
    for (const auto& s : sp.samples) {
        re_sum += s.real();
        im_sum += s.imag();
        re_sum_sq += s.real() * s.real();
        im_sum_sq += s.imag() * s.imag();
        re_im_sum += s.real() * s.imag();
    }
    const auto tc = static_cast<double>(sp.total_count);
    const auto re_var = ((re_sum_sq - re_sum * re_sum / tc) / (tc - 1)).eval();
    const auto im_var = ((im_sum_sq - im_sum * im_sum / tc) / (tc - 1)).eval();
    const auto re_im_cov = ((re_im_sum - re_sum * im_sum / tc) / (tc - 1)).eval();
    return std::make_tuple(re_var, im_var, re_im_cov);
}

// Calculate the analytic covariance of the given stochastic process.
template <typename S>
    requires std::is_same_v<typename S::value_type, double>
[[nodiscard]] auto analytic_covariance(const S& sp) {
    auto cov = S::mat_type::Zero().eval();
    const auto mean = analytic_mean(sp);
    for (const auto& [s, p] : simplemc::ranges::views::zip(sp.states, sp.probs)) {
        cov += (p * (s - mean).matrix() * (s - mean).matrix().transpose()).array();
    }
    return cov;
}

template <typename S>
    requires std::is_same_v<typename S::value_type, std::complex<double>>
[[nodiscard]] auto analytic_covariance(const S& sp) {
    auto re_cov = S::dbl_mat_type::Zero().eval();
    auto im_cov = S::dbl_mat_type::Zero().eval();
    auto re_im_cov = S::dbl_mat_type::Zero().eval();
    const auto mean = analytic_mean(sp);
    for (const auto& [s, p] : simplemc::ranges::views::zip(sp.states, sp.probs)) {
        auto tmp = s - mean;
        re_cov += (p * tmp.real().matrix() * tmp.real().matrix().transpose()).array();
        im_cov += (p * tmp.imag().matrix() * tmp.imag().matrix().transpose()).array();
        re_im_cov += (p * tmp.real().matrix() * tmp.imag().matrix().transpose()).array();
    }
    return std::make_tuple(re_cov, im_cov, re_im_cov);
}

// Calculate the sample covariance of the given stochastic process.
template <typename S>
    requires std::is_same_v<typename S::value_type, double>
[[nodiscard]] auto sample_covariance(const S& sp) {
    auto sum = S::vec_type::Zero().eval();
    auto sum_sq = S::mat_type::Zero().eval();
    for (const auto& s : sp.samples) {
        sum += s;
        sum_sq += (s.matrix() * s.matrix().transpose()).array();
    }
    const auto tc = static_cast<double>(sp.total_count);
    return ((sum_sq - (sum.matrix() * sum.matrix().transpose()).array() / tc) / (tc - 1)).eval();
}

template <typename S>
    requires std::is_same_v<typename S::value_type, std::complex<double>>
[[nodiscard]] auto sample_covariance(const S& sp) {
    auto re_sum = S::dbl_vec_type::Zero().eval();
    auto im_sum = S::dbl_vec_type::Zero().eval();
    auto re_sum_sq = S::dbl_mat_type::Zero().eval();
    auto im_sum_sq = S::dbl_mat_type::Zero().eval();
    auto re_im_sum = S::dbl_mat_type::Zero().eval();
    for (const auto& s : sp.samples) {
        re_sum += s.real();
        im_sum += s.imag();
        re_sum_sq += (s.real().matrix() * s.real().matrix().transpose()).array();
        im_sum_sq += (s.imag().matrix() * s.imag().matrix().transpose()).array();
        re_im_sum += (s.real().matrix() * s.imag().matrix().transpose()).array();
    }
    const auto tc = static_cast<double>(sp.total_count);
    const auto re_var = ((re_sum_sq - (re_sum.matrix() * re_sum.matrix().transpose()).array() / tc) / (tc - 1)).eval();
    const auto im_var = ((im_sum_sq - (im_sum.matrix() * im_sum.matrix().transpose()).array() / tc) / (tc - 1)).eval();
    const auto re_im_cov =
        ((re_im_sum - (re_sum.matrix() * im_sum.matrix().transpose()).array() / tc) / (tc - 1)).eval();
    return std::make_tuple(re_var, im_var, re_im_cov);
}

// Block the samples of a stochastic process.
template <typename S>
[[nodiscard]] auto block_samples(const S& sp, std::uint64_t blsize) {
    S bsp;
    bsp.states = sp.states;
    bsp.probs = sp.probs;
    bsp.total_count = sp.total_count / blsize;
    bsp.samples.resize(bsp.total_count);
    for (std::uint64_t i = 0; i < bsp.total_count; ++i) {
        auto sum = S::state_type::Zero().eval();
        for (std::uint64_t j = 0; j < blsize; ++j) {
            sum += sp.samples[i * blsize + j];
        }
        bsp.samples[i] = sum / static_cast<double>(blsize);
    }
    return bsp;
}

// Calculate the sample autocorrelations and the integrated autocorrelation time (only for double).
template <typename S>
    requires std::is_same_v<typename S::value_type, double>
[[nodiscard]] auto sample_autocorr(const S& sp, std::uint64_t num = 100) {
    const auto tc = static_cast<double>(sp.total_count);
    const auto mean = analytic_mean(sp);
    const auto var = analytic_variance(sp);
    auto taus = std::vector<typename S::vec_type>(num);
    auto tau_int = S::vec_type::Constant(0.0).eval();
    for (std::uint64_t i = 0; i < num; ++i) {
        const auto id = static_cast<double>(i + 1);
        for (std::uint64_t j = 0; j < sp.total_count - (i + 1); ++j) {
            taus[i] += (sp.samples[j] - mean) * (sp.samples[j + i + 1] - mean);
        }
        taus[i] /= ((tc - id) * var);
        tau_int += taus[i] * (1.0 - id / tc);
    }
    return std::make_tuple(taus, tau_int);
}

// Estimate autocorrelation using blocking method (only for double).
template <typename S>
    requires std::is_same_v<typename S::value_type, double>
[[nodiscard]] auto blocking_autocorr(const S& sp, std::uint64_t fac = 2) {
    auto bsps = std::vector<S> {};
    auto taus_v = std::vector<typename S::vec_type> {};
    auto taus_c = std::vector<typename S::mat_type> {};
    bsps.emplace_back(sp);
    const auto v0 = sample_variance(bsps[0]);
    taus_v.emplace_back(0.0);
    const auto c0 = sample_covariance(bsps[0]);
    taus_c.emplace_back(0.0);
    auto size = fac;
    while (size < sp.total_count) {
        bsps.emplace_back(block_samples(sp, size));
        const auto vi = sample_variance(bsps.back());
        taus_v.emplace_back(0.5 * ((vi * size) / v0 - 1));
        const auto ci = sample_covariance(bsps.back());
        taus_c.emplace_back(0.5 * ((ci * size) / c0 - 1));
        size *= fac;
    }
    return std::make_tuple(bsps, taus_v, taus_c);
}

#endif // SIMPLEMC_TESTS_ACCS_STOCHASTIC_PROCESS_HPP
