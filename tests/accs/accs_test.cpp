/**
 * @file accs_test.cpp
 * @brief Unit tests for simplemc-accs.
 */

#include "./stochastic_process.hpp"
#include "../test_utils.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/utils/format.hpp>

#include <fmt/ranges.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

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

        auto s1_d = state_d{ 1.0, 0.0, 0.0 };
        auto s2_d = state_d{ 0.0, 1.0, 0.0 };
        auto s3_d = state_d{ 0.0, 0.0, 1.0 };
        auto probs_d = std::vector<double>{ 1.0, 1.0, 1.0 };
        sp_d.set_states(std::vector<state_d>{ s1_d, s2_d, s3_d });
        sp_d.set_weights(probs_d);
        sp_d.run(steps);

        auto s1_c = state_c{ 1.0 - 0.5i , 0.0 + 0.0i , 0.0 + 0.0i }; 
        auto s2_c = state_c{ 0.0 + 0.0i, -0.3 + 0.8i, 0.0 + 0.0i }; 
        auto s3_c = state_c{ 0.0 + 0.0i, 0.0 + 0.0i, 0.9 + 3.4i };
        auto probs_c = std::vector<double>{ 1.0, 3.4, 0.6 };;
        sp_c.set_states(std::vector<state_c>{ s1_c, s2_c, s3_c });
        sp_c.set_weights(probs_c);
        sp_c.run(steps);
    }

    proc_d sp_d;
    proc_c sp_c;
    int steps { 100000 };
};

// Fill accumulators with random samples from a normal distribution.
template <typename A, typename T>
void fill_accs(A& acc_sv, A& acc_mva, A& acc_rg, const std::vector<T>& samples) {
    const auto size = acc_mva.size();
    for (std::size_t i = 0; i < samples.size(); ++i) {
        auto mva = acc_mva.make_mva();
        acc_sv << samples[i](0);
        for (int j = 0; j < size; ++j) {
            mva[j] << samples[i](j);
        }
        acc_rg.accumulate(samples[i]);
    }
}

// Check empty accumulator.
template <typename A>
void check_empty(const A& acc) {
    auto check = [](auto val) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(val)>, double>) {
            ASSERT_TRUE(std::isnan(val));
        } else {
            ASSERT_TRUE(std::isnan(val.real()));
            ASSERT_TRUE(std::isnan(val.imag()));
        }
    };
    ASSERT_EQ(acc.count(), 0);
    auto mean = acc.mean();
    for (int i = 0; i < acc.size(); ++i) {
        check(mean[i]);
    }
}

// Print analytic resutls of the telegraph processes.
TEST_F(SimplemcAccs, PrintAnalyticResults) {
    fmt::print("Expected results for stochastic process with double states:\n");
    fmt::print("State space: \n", sp_d.states);
    fmt::print("Stationary distribution: {}\n", sp_d.probs);
    fmt::print("Analytic mean: {}\n", analytic_mean(sp_d));
    fmt::print("Sample mean: {}\n", sample_mean(sp_d));
    fmt::print("Analytic variance: {}\n", analytic_variance(sp_d));
    fmt::print("Sample variance: {}\n", sample_variance(sp_d));

    const auto [a_re_var, a_im_var, a_re_im_cov] = analytic_variance(sp_c);
    const auto [s_re_var, s_im_var, s_re_im_cov] = sample_variance(sp_c);
    fmt::print("\nExpected results for stochastic process with complex states:\n");
    fmt::print("State space: \n", sp_c.states);
    fmt::print("Stationary distribution: {}\n", sp_c.probs);
    fmt::print("Analytic mean: {}\n", analytic_mean(sp_c));
    fmt::print("Sample mean: {}\n", sample_mean(sp_c));
    fmt::print("Real analytic variance: {}\n", a_re_var);
    fmt::print("Real sample variance: {}\n", s_re_var);
    fmt::print("Imaginary analytic variance: {}\n", a_im_var);
    fmt::print("Imaginary sample variance: {}\n", s_im_var);
    fmt::print("Real-imaginary analytic covariance (diagonal): {}\n", a_re_im_cov);
    fmt::print("Real-imaginary sample covariance (diagonal): {}\n", s_re_im_cov);
}

// Test mean accumulator.
TEST_F(SimplemcAccs, MeanAccumulator) {
    // general set up
    using acc_d = simplemc::mean_acc<double>;
    using acc_c = simplemc::mean_acc<std::complex<double>, simplemc::accs::varalg::welford>;
    using storage_d = typename acc_d::storage_type;
    using storage_c = typename acc_c::storage_type;
    double tol = 1e-2;
    acc_d acc_sv_d(1, 5.0), acc_mva_d(size, 2.0), acc_rg_d(size, 1.0);
    acc_c acc_sv_c(1, { 5.0, 1.0 }), acc_mva_c(size, { -3.2, -2.2 }), acc_rg_c(size, 10.0);

    // check size of empty accumulators
    ASSERT_EQ(acc_sv_d.size(), 1);
    ASSERT_EQ(acc_mva_d.size(), size);
    ASSERT_EQ(acc_rg_d.size(), size);
    ASSERT_EQ(acc_sv_c.size(), 1);
    ASSERT_EQ(acc_mva_c.size(), size);
    ASSERT_EQ(acc_rg_c.size(), size);

    // check empty accumulators
    check_empty(acc_sv_d);
    check_empty(acc_mva_d);
    check_empty(acc_rg_d);
    check_empty(acc_sv_c);
    check_empty(acc_mva_c);
    check_empty(acc_rg_c);

    // fill accumulators
    fill_accs(acc_sv_d, acc_mva_d, acc_rg_d, sp_d.samples);
    fill_accs(acc_sv_c, acc_mva_c, acc_rg_c, sp_c.samples);

    // check size of filled accumulators
    ASSERT_EQ(acc_sv_d.count(), sp_d.total_count);
    ASSERT_EQ(acc_mva_d.count(), sp_d.total_count);
    ASSERT_EQ(acc_rg_d.count(), sp_d.total_count);
    ASSERT_EQ(acc_sv_c.count(), sp_c.total_count);
    ASSERT_EQ(acc_mva_c.count(), sp_c.total_count);
    ASSERT_EQ(acc_rg_c.count(), sp_c.total_count);

    // check mean
    const auto m_d = sample_mean(sp_d);
    const auto m_c = sample_mean(sp_c);
    check_near(acc_sv_d.mean()[0], m_d[0], tol);
    check_range_near(acc_mva_d.mean(), m_d, tol);
    check_range_near(acc_rg_d.mean(), m_d, tol);
    check_near(acc_sv_c.mean()[0], m_c[0], tol);
    check_range_near(acc_mva_c.mean(), m_c, tol);
    check_range_near(acc_rg_c.mean(), m_c, tol);

    // check reset and merging of accumulators
    acc_d merge_d(storage_d::Constant(size, 2.0).eval());
    acc_c merge_c(storage_c::Constant(size, { 2.0, 1.0 }).eval());
    ASSERT_EQ(merge_d.count(), 0);
    ASSERT_EQ(merge_c.count(), 0);
    merge_d << acc_mva_d;
    merge_d << acc_rg_d;
    merge_c << acc_mva_c;
    merge_c << acc_rg_c;
    ASSERT_EQ(merge_d.count(), 2 * sp_d.total_count);
    ASSERT_EQ(merge_c.count(), 2 * sp_c.total_count);
    check_range_near(merge_d.mean(), m_d, tol);
    check_range_near(merge_c.mean(), m_c, tol);
}

// Test variance accumulator for double values.
TEST_F(SimplemcAccs, DoubleVarianceAccumulator) {
    // general set up
    using acc_std = simplemc::var_acc<double, simplemc::accs::varalg::standard>;
    using acc_wel = simplemc::var_acc<double, simplemc::accs::varalg::welford>;
    using storage_d = typename acc_std::storage_type;
    double tol = 1e-2;
    acc_std acc_sv_std(1, 5.0), acc_mva_std(size, 2.0), acc_rg_std(size, 1.0);
    acc_wel acc_sv_wel(1, 5.0), acc_mva_wel(size, 2.0), acc_rg_wel(size, 1.0);

    // check size of empty accumulators
    ASSERT_EQ(acc_sv_std.size(), 1);
    ASSERT_EQ(acc_mva_std.size(), size);
    ASSERT_EQ(acc_rg_std.size(), size);
    ASSERT_EQ(acc_sv_wel.size(), 1);
    ASSERT_EQ(acc_mva_wel.size(), size);
    ASSERT_EQ(acc_rg_wel.size(), size);

    // check empty accumulators
    check_empty(acc_sv_std);
    check_empty(acc_mva_std);
    check_empty(acc_rg_std);
    check_empty(acc_sv_wel);
    check_empty(acc_mva_wel);
    check_empty(acc_rg_wel);

    // fill accumulators
    fill_accs(acc_sv_std, acc_mva_std, acc_rg_std, sp_d.samples);
    fill_accs(acc_sv_wel, acc_mva_wel, acc_rg_wel, sp_d.samples);

    // check size of filled accumulators
    ASSERT_EQ(acc_sv_std.count(), sp_d.total_count);
    ASSERT_EQ(acc_mva_std.count(), sp_d.total_count);
    ASSERT_EQ(acc_rg_std.count(), sp_d.total_count);
    ASSERT_EQ(acc_sv_wel.count(), sp_d.total_count);
    ASSERT_EQ(acc_mva_wel.count(), sp_d.total_count);
    ASSERT_EQ(acc_rg_wel.count(), sp_d.total_count);

    // check mean
    const auto m_d = sample_mean(sp_d);
    check_near(acc_sv_std.mean()[0], m_d[0], tol);
    check_range_near(acc_mva_std.mean(), m_d, tol);
    check_range_near(acc_rg_std.mean(), m_d, tol);
    check_near(acc_sv_wel.mean()[0], m_d[0], tol);
    check_range_near(acc_mva_wel.mean(), m_d, tol);
    check_range_near(acc_rg_wel.mean(), m_d, tol);

    // check stderror/variance
    const auto v_d = sample_variance(sp_d);
    check_near(acc_sv_std.variance_of_data()[0], v_d[0], tol);
    check_range_near(acc_mva_std.variance_of_data(), v_d, tol);
    check_range_near(acc_rg_std.variance_of_data(), v_d, tol);
    check_near(acc_sv_wel.variance_of_data()[0], v_d[0], tol);
    check_range_near(acc_mva_wel.variance_of_data(), v_d, tol);
    check_range_near(acc_rg_wel.variance_of_data(), v_d, tol);

    // check reset and merging of accumulators
    acc_std merge_std(storage_d::Constant(size, -12.7).eval());
    acc_wel merge_wel(storage_d::Constant(size, -12.7).eval());
    ASSERT_EQ(merge_std.count(), 0);
    ASSERT_EQ(merge_wel.count(), 0);
    merge_std << acc_mva_std;
    merge_std << acc_rg_std;
    merge_wel << acc_mva_wel;
    merge_wel << acc_rg_wel;
    ASSERT_EQ(merge_std.count(), 2 * sp_d.total_count);
    ASSERT_EQ(merge_wel.count(), 2 * sp_d.total_count);
    check_range_near(merge_std.mean(), m_d, tol);
    check_range_near(merge_std.variance_of_data(), v_d, tol);
    check_range_near(merge_wel.mean(), m_d, tol);
    check_range_near(merge_wel.variance_of_data(), v_d, tol);
}

// Test variance accumulator for complex values.
TEST_F(SimplemcAccs, ComplexVarianceAccumulator) {
    // general set up
    using acc_std = simplemc::var_acc<std::complex<double>, simplemc::accs::varalg::standard>;
    using acc_wel = simplemc::var_acc<std::complex<double>, simplemc::accs::varalg::welford>;
    using storage_c = typename acc_std::storage_type;
    double tol = 1e-2;
    acc_std acc_sv_std(1, { 5.0, 1.0 }), acc_mva_std(size, { -3.2, -2.2 }), acc_rg_std(size, 10.0);
    acc_wel acc_sv_wel(1, { 5.0, 1.0 }), acc_mva_wel(size, { -3.2, -2.2 }), acc_rg_wel(size, 10.0);

    // check size of empty accumulators
    ASSERT_EQ(acc_sv_std.size(), 1);
    ASSERT_EQ(acc_mva_std.size(), size);
    ASSERT_EQ(acc_rg_std.size(), size);
    ASSERT_EQ(acc_sv_wel.size(), 1);
    ASSERT_EQ(acc_mva_wel.size(), size);
    ASSERT_EQ(acc_rg_wel.size(), size);

    // check empty accumulators
    check_empty(acc_sv_std);
    check_empty(acc_mva_std);
    check_empty(acc_rg_std);
    check_empty(acc_sv_wel);
    check_empty(acc_mva_wel);
    check_empty(acc_rg_wel);

    // fill accumulators
    fill_accs(acc_sv_std, acc_mva_std, acc_rg_std, sp_c.samples);
    fill_accs(acc_sv_wel, acc_mva_wel, acc_rg_wel, sp_c.samples);

    // check size of filled accumulators
    ASSERT_EQ(acc_sv_std.count(), sp_c.total_count);
    ASSERT_EQ(acc_mva_std.count(), sp_c.total_count);
    ASSERT_EQ(acc_rg_std.count(), sp_c.total_count);
    ASSERT_EQ(acc_sv_wel.count(), sp_c.total_count);
    ASSERT_EQ(acc_mva_wel.count(), sp_c.total_count);
    ASSERT_EQ(acc_rg_wel.count(), sp_c.total_count);

    // check mean
    const auto m_c = sample_mean(sp_c);
    check_near(acc_sv_std.mean()[0], m_c[0], tol);
    check_range_near(acc_mva_std.mean(), m_c, tol);
    check_range_near(acc_rg_std.mean(), m_c, tol);
    check_near(acc_sv_wel.mean()[0], m_c[0], tol);
    check_range_near(acc_mva_wel.mean(), m_c, tol);
    check_range_near(acc_rg_wel.mean(), m_c, tol);

    // check stderror/variance
    const auto [re_v, im_v, reim_v] = sample_variance(sp_c);

    check_near(acc_sv_std.variance_of_real_data()[0], re_v[0], tol);
    check_range_near(acc_mva_std.variance_of_real_data(), re_v, tol);
    check_range_near(acc_rg_std.variance_of_real_data(), re_v, tol);
    check_near(acc_sv_wel.variance_of_real_data()[0], re_v[0], tol);
    check_range_near(acc_mva_wel.variance_of_real_data(), re_v, tol);
    check_range_near(acc_rg_wel.variance_of_real_data(), re_v, tol);

    check_near(acc_sv_std.variance_of_imag_data()[0], im_v[0], tol);
    check_range_near(acc_mva_std.variance_of_imag_data(), im_v, tol);
    check_range_near(acc_rg_std.variance_of_imag_data(), im_v, tol);
    check_near(acc_sv_wel.variance_of_imag_data()[0], im_v[0], tol);
    check_range_near(acc_mva_wel.variance_of_imag_data(), im_v, tol);
    check_range_near(acc_rg_wel.variance_of_imag_data(), im_v, tol);

    check_near(acc_sv_std.covariance_of_real_and_imag_data()[0], reim_v[0], tol);
    check_range_near(acc_mva_std.covariance_of_real_and_imag_data(), reim_v, tol);
    check_range_near(acc_rg_std.covariance_of_real_and_imag_data(), reim_v, tol);
    check_near(acc_sv_wel.covariance_of_real_and_imag_data()[0], reim_v[0], tol);
    check_range_near(acc_mva_wel.covariance_of_real_and_imag_data(), reim_v, tol);
    check_range_near(acc_rg_wel.covariance_of_real_and_imag_data(), reim_v, tol);

    // check reset and merging of accumulators
    acc_std merge_std(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    acc_wel merge_wel(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    ASSERT_EQ(merge_std.count(), 0);
    ASSERT_EQ(merge_wel.count(), 0);
    merge_std << acc_mva_std;
    merge_std << acc_rg_std;
    merge_wel << acc_mva_wel;
    merge_wel << acc_rg_wel;
    ASSERT_EQ(merge_std.count(), 2 * sp_c.total_count);
    ASSERT_EQ(merge_wel.count(), 2 * sp_c.total_count);
    check_range_near(merge_std.mean(), m_c, tol);
    check_range_near(merge_std.covariance_of_real_and_imag_data(), reim_v, tol);
    check_range_near(merge_wel.mean(), m_c, tol);
    check_range_near(merge_wel.covariance_of_real_and_imag_data(), reim_v, tol);
}