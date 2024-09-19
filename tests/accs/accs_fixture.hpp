#ifndef SIMPLEMC_TESTS_ACCS_FIXTURE_HPP
#define SIMPLEMC_TESTS_ACCS_FIXTURE_HPP

#include "./stochastic_process.hpp"
#include "../test_utils.hpp"

#include <simplemc/utils/fmt_complex.hpp>
#include <simplemc/utils/to_string.hpp>

#include <Eigen/Dense>
#include <fmt/ranges.h>

#include <complex>

// Test fixture for the accumulators.
class SimplemcAccs : public ::testing::Test {
protected:
    constexpr static long size = 3;
    using proc_d = stochastic_process<double, size>;
    using proc_c = stochastic_process<std::complex<double>, size>;
    using state_d = typename proc_d::state_type;
    using state_c = typename proc_c::state_type;

    void SetUp() override {
        using namespace std::literals::complex_literals;

        auto s1_d = state_d { 1.0, 0.0, 0.0 };
        auto s2_d = state_d { 0.0, 1.0, 0.0 };
        auto s3_d = state_d { 0.0, 0.0, 1.0 };
        auto probs_d = std::vector<double> { 2.5, 0.3, 1.5 };
        Eigen::MatrixXd mcmc_props_d(3, 3);
        mcmc_props_d << 10, 1, 3, 2, 8, 1, 4, 1, 10;
        sp_d.set_states(std::vector<state_d> { s1_d, s2_d, s3_d });
        sp_d.set_weights(probs_d);
        sp_d.set_mcmc_proposal(mcmc_props_d);
        sp_d.mcmc_warmup(steps);
        sp_d.mcmc_run(steps);

        auto s1_c = state_c { 1.0 - 0.5i, 0.0 + 0.0i, 0.0 + 0.0i };
        auto s2_c = state_c { 0.0 + 0.0i, -0.3 + 0.8i, 0.0 + 0.0i };
        auto s3_c = state_c { 0.0 + 0.0i, 0.0 + 0.0i, 0.9 + 3.4i };
        auto probs_c = std::vector<double> { 1.0, 3.4, 0.6 };
        Eigen::MatrixXd mcmc_props_c(3, 3);
        mcmc_props_c << 10, 3, 9, 8, 1, 2, 4, 4, 7;
        sp_c.set_states(std::vector<state_c> { s1_c, s2_c, s3_c });
        sp_c.set_weights(probs_c);
        sp_c.set_mcmc_proposal(mcmc_props_c);
        sp_c.mcmc_warmup(steps);
        sp_c.mcmc_run(steps);
    }

    void print_analytic_results() const {
        fmt::print("Expected results for stochastic process with double states:\n");
        fmt::print("State space: \n", sp_d.states);
        fmt::print("Stationary distribution: {}\n", sp_d.probs);
        fmt::print("Analytic mean: {}\n", analytic_mean(sp_d));
        fmt::print("Sample mean: {}\n", sample_mean(sp_d));
        fmt::print("Analytic variance: {}\n", analytic_variance(sp_d));
        fmt::print("Sample variance: {}\n", sample_variance(sp_d));
        fmt::print("Analytic covariance:\n{}\n", simplemc::to_string(analytic_covariance(sp_d)));
        fmt::print("Sample covariance:\n{}\n", simplemc::to_string(sample_covariance(sp_d)));
        fmt::print("Integrated tau: {}\n", std::get<1>(sample_autocorr(sp_d, 100)));

        const auto [a_re_var, a_im_var, a_re_im_var] = analytic_variance(sp_c);
        const auto [s_re_var, s_im_var, s_re_im_var] = sample_variance(sp_c);
        const auto [a_re_cov, a_im_cov, a_re_im_cov] = analytic_covariance(sp_c);
        const auto [s_re_cov, s_im_cov, s_re_im_cov] = sample_covariance(sp_c);
        fmt::print("\nExpected results for stochastic process with complex states:\n");
        fmt::print("State space: \n", sp_c.states);
        fmt::print("Stationary distribution: {}\n", sp_c.probs);
        fmt::print("Analytic mean: {}\n", analytic_mean(sp_c));
        fmt::print("Sample mean: {}\n", sample_mean(sp_c));
        fmt::print("Real analytic variance: {}\n", a_re_var);
        fmt::print("Real sample variance: {}\n", s_re_var);
        fmt::print("Imaginary analytic variance: {}\n", a_im_var);
        fmt::print("Imaginary sample variance: {}\n", s_im_var);
        fmt::print("Real-imaginary analytic covariance (diagonal): {}\n", a_re_im_var);
        fmt::print("Real-imaginary sample covariance (diagonal): {}\n", s_re_im_var);
        fmt::print("Real analytic covariance:\n{}\n", simplemc::to_string(a_re_cov));
        fmt::print("Real sample covariance:\n{}\n", simplemc::to_string(s_re_cov));
        fmt::print("Imaginary analytic covariance:\n{}\n", simplemc::to_string(a_im_cov));
        fmt::print("Imaginary sample covariance:\n{}\n", simplemc::to_string(s_im_cov));
        fmt::print("Real-imaginary analytic covariance:\n{}\n", simplemc::to_string(a_re_im_cov));
        fmt::print("Real-imaginary sample covariance:\n{}\n", simplemc::to_string(s_re_im_cov));
    }

    proc_d sp_d;
    proc_c sp_c;
    int steps { 100000 };
};

// Check an empty accumulator.
template <typename A>
void check_empty(const A& acc) {
    ASSERT_EQ(acc.count(), 0);
    auto mean = acc.mean();
    if constexpr (!A::is_dynamic && A::static_size == 1) {
        check_isnan(mean);
    } else {
        for (int i = 0; i < acc.size(); ++i) {
            check_isnan(mean[i]);
        }
    }
}

#endif // SIMPLEMC_TESTS_ACCS_FIXTURE_HPP
