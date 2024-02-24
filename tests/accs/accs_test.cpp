/**
 * @file accs_test.cpp
 * @brief Unit tests for simplemc-accs.
 */

#include "./telegraph_process.hpp"
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
    void SetUp() override {
        tp_d.set_states(-2.0, 6.0);
        tp_d.set_probabilities(0.123, 0.3298);
        tp_d.warm_up(10000);
        tp_d.simulate(steps);

        tp_c.set_states(std::complex<double> { -1.2, 2.4 }, std::complex<double> { 4.5, 3.2 });
        tp_c.set_probabilities(0.8, 0.2);
        tp_c.warm_up(10000);
        tp_c.simulate(steps);
    }

    telegraph_process<double> tp_d;
    telegraph_process<std::complex<double>> tp_c;
    int steps { 100000 };
};

// Fill accumulators with random samples from a normal distribution.
template <typename A, typename T>
void fill_accs(A& acc_sv, A& acc_mva, A& acc_rg, const std::vector<T>& samples) {
    const auto size = acc_mva.size();
    std::vector<T> vec(size);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        auto mva = acc_mva.make_mva();
        acc_sv << samples[i];
        for (int j = 0; j < size; ++j) {
            mva[j] << samples[i];
            vec[j] = samples[i];
        }
        acc_rg.accumulate(vec);
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
    fmt::print("Analytic results for telegraph process with double states:\n");
    fmt::print("State 1: {}, State 2: {}\n", tp_d.states[0], tp_d.states[1]);
    fmt::print("Transition probabilities: {}, {}\n", tp_d.probs[0], tp_d.probs[1]);
    fmt::print("Mean: {}\n", tp_d.mean());
    fmt::print("Variance: {}\n", tp_d.variance());
    fmt::print("Stationary distribution: {}\n", tp_d.stationary_distribution());

    const auto [re_var, im_var, re_im_cov] = tp_c.variance();
    fmt::print("\nAnalytic results for telegraph process with complex states:\n");
    fmt::print("State 1: {}, State 2: {}\n", tp_c.states[0], tp_c.states[1]);
    fmt::print("Transition probabilities: {}, {}\n", tp_c.probs[0], tp_c.probs[1]);
    fmt::print("Mean: {}\n", tp_c.mean());
    fmt::print("Real variance: {}\n", re_var);
    fmt::print("Imaginary variance: {}\n", im_var);
    fmt::print("Real-imaginary covariance: {}\n", re_im_cov);
    fmt::print("Stationary distribution: {}\n", tp_c.stationary_distribution());
}

// Test mean accumulator.
TEST_F(SimplemcAccs, MeanAccumulator) {
    // general set up
    using acc_d = simplemc::mean_acc<double>;
    using acc_c = simplemc::mean_acc<std::complex<double>, simplemc::accs::varalg::welford>;
    using storage_d = typename acc_d::storage_type;
    using storage_c = typename acc_c::storage_type;
    double tol = 1e-2;
    int size = 2;
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
    const auto samples_d = tp_d.get_sampled_values();
    const auto samples_c = tp_c.get_sampled_values();
    fill_accs(acc_sv_d, acc_mva_d, acc_rg_d, samples_d);
    fill_accs(acc_sv_c, acc_mva_c, acc_rg_c, samples_c);

    // check size of filled accumulators
    ASSERT_EQ(acc_sv_d.count(), tp_d.total_count);
    ASSERT_EQ(acc_mva_d.count(), tp_d.total_count);
    ASSERT_EQ(acc_rg_d.count(), tp_d.total_count);
    ASSERT_EQ(acc_sv_c.count(), tp_c.total_count);
    ASSERT_EQ(acc_mva_c.count(), tp_c.total_count);
    ASSERT_EQ(acc_rg_c.count(), tp_c.total_count);

    // check mean
    const auto m_d = sample_mean(samples_d);
    const auto m_vec_d = storage_d::Constant(size, m_d);
    const auto m_c = sample_mean(samples_c);
    const auto m_vec_c = storage_c::Constant(size, m_c);
    check_near(acc_sv_d.mean()[0], m_d, tol);
    check_range_near(acc_mva_d.mean(), m_vec_d, tol);
    check_range_near(acc_rg_d.mean(), m_vec_d, tol);
    check_near(acc_sv_c.mean()[0], m_c, tol);
    check_range_near(acc_mva_c.mean(), m_vec_c, tol);
    check_range_near(acc_rg_c.mean(), m_vec_c, tol);

    // check reset and merging of accumulators
    acc_d merge_d(storage_d::Constant(size, 2.0).eval());
    acc_c merge_c(storage_c::Constant(size, { 2.0, 1.0 }).eval());
    ASSERT_EQ(merge_d.count(), 0);
    ASSERT_EQ(merge_c.count(), 0);
    merge_d << acc_mva_d;
    merge_d << acc_rg_d;
    merge_c << acc_mva_c;
    merge_c << acc_rg_c;
    ASSERT_EQ(merge_d.count(), 2 * tp_d.total_count);
    ASSERT_EQ(merge_c.count(), 2 * tp_c.total_count);
    check_range_near(merge_d.mean(), m_vec_d, tol);
    check_range_near(merge_c.mean(), m_vec_c, tol);
}

// Test variance accumulator for double values.
TEST_F(SimplemcAccs, DoubleVarianceAccumulator) {
    // general set up
    using acc_std = simplemc::var_acc<double, simplemc::accs::varalg::standard>;
    using acc_wel = simplemc::var_acc<double, simplemc::accs::varalg::welford>;
    using storage_d = typename acc_std::storage_type;
    double tol = 1e-2;
    int size = 2;
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
    const auto samples_d = tp_d.get_sampled_values();
    fill_accs(acc_sv_std, acc_mva_std, acc_rg_std, samples_d);
    fill_accs(acc_sv_wel, acc_mva_wel, acc_rg_wel, samples_d);

    // check size of filled accumulators
    ASSERT_EQ(acc_sv_std.count(), tp_d.total_count);
    ASSERT_EQ(acc_mva_std.count(), tp_d.total_count);
    ASSERT_EQ(acc_rg_std.count(), tp_d.total_count);
    ASSERT_EQ(acc_sv_wel.count(), tp_d.total_count);
    ASSERT_EQ(acc_mva_wel.count(), tp_d.total_count);
    ASSERT_EQ(acc_rg_wel.count(), tp_d.total_count);

    // check mean
    const auto m_d = sample_mean(samples_d);
    const auto m_vec_d = storage_d::Constant(size, m_d);
    check_near(acc_sv_std.mean()[0], m_d, tol);
    check_range_near(acc_mva_std.mean(), m_vec_d, tol);
    check_range_near(acc_rg_std.mean(), m_vec_d, tol);
    check_near(acc_sv_wel.mean()[0], m_d, tol);
    check_range_near(acc_mva_wel.mean(), m_vec_d, tol);
    check_range_near(acc_rg_wel.mean(), m_vec_d, tol);

    // check stderror/variance
    const auto v_d = sample_variance(samples_d);
    const auto v_vec_d = storage_d::Constant(size, v_d);
    check_near(acc_sv_std.variance_of_data()[0], v_d, tol);
    check_range_near(acc_mva_std.variance_of_data(), v_vec_d, tol);
    check_range_near(acc_rg_std.variance_of_data(), v_vec_d, tol);
    check_near(acc_sv_wel.variance_of_data()[0], v_d, tol);
    check_range_near(acc_mva_wel.variance_of_data(), v_vec_d, tol);
    check_range_near(acc_rg_wel.variance_of_data(), v_vec_d, tol);

    // check reset and merging of accumulators
    acc_std merge_std(storage_d::Constant(size, -12.7).eval());
    acc_wel merge_wel(storage_d::Constant(size, -12.7).eval());
    ASSERT_EQ(merge_std.count(), 0);
    ASSERT_EQ(merge_wel.count(), 0);
    merge_std << acc_mva_std;
    merge_std << acc_rg_std;
    merge_wel << acc_mva_wel;
    merge_wel << acc_rg_wel;
    ASSERT_EQ(merge_std.count(), 2 * tp_d.total_count);
    ASSERT_EQ(merge_wel.count(), 2 * tp_d.total_count);
    check_range_near(merge_std.mean(), m_vec_d, tol);
    check_range_near(merge_std.variance_of_data(), v_vec_d, tol);
    check_range_near(merge_wel.mean(), m_vec_d, tol);
    check_range_near(merge_wel.variance_of_data(), v_vec_d, tol);
}

// Test variance accumulator for complex values.
TEST_F(SimplemcAccs, ComplexVarianceAccumulator) {
    // general set up
    using acc_std = simplemc::var_acc<std::complex<double>, simplemc::accs::varalg::standard>;
    using acc_wel = simplemc::var_acc<std::complex<double>, simplemc::accs::varalg::welford>;
    using storage_c = typename acc_std::storage_type;
    using storage_d = typename acc_wel::dbl_storage_type;
    double tol = 1e-2;
    int size = 2;
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
    const auto samples_c = tp_c.get_sampled_values();
    fill_accs(acc_sv_std, acc_mva_std, acc_rg_std, samples_c);
    fill_accs(acc_sv_wel, acc_mva_wel, acc_rg_wel, samples_c);

    // check size of filled accumulators
    ASSERT_EQ(acc_sv_std.count(), tp_c.total_count);
    ASSERT_EQ(acc_mva_std.count(), tp_c.total_count);
    ASSERT_EQ(acc_rg_std.count(), tp_c.total_count);
    ASSERT_EQ(acc_sv_wel.count(), tp_c.total_count);
    ASSERT_EQ(acc_mva_wel.count(), tp_c.total_count);
    ASSERT_EQ(acc_rg_wel.count(), tp_c.total_count);

    // check mean
    const auto m_c = sample_mean(samples_c);
    const auto m_vec_c = storage_c::Constant(size, m_c);
    check_near(acc_sv_std.mean()[0], m_c, tol);
    check_range_near(acc_mva_std.mean(), m_vec_c, tol);
    check_range_near(acc_rg_std.mean(), m_vec_c, tol);
    check_near(acc_sv_wel.mean()[0], m_c, tol);
    check_range_near(acc_mva_wel.mean(), m_vec_c, tol);
    check_range_near(acc_rg_wel.mean(), m_vec_c, tol);

    // check stderror/variance
    const auto [re_v, im_v, reim_v] = sample_variance(samples_c);
    const auto re_v_vec = storage_d::Constant(size, re_v);
    const auto im_v_vec = storage_d::Constant(size, im_v);
    const auto reim_v_vec = storage_d::Constant(size, reim_v);

    check_near(acc_sv_std.variance_of_real_data()[0], re_v, tol);
    check_range_near(acc_mva_std.variance_of_real_data(), re_v_vec, tol);
    check_range_near(acc_rg_std.variance_of_real_data(), re_v_vec, tol);
    check_near(acc_sv_wel.variance_of_real_data()[0], re_v, tol);
    check_range_near(acc_mva_wel.variance_of_real_data(), re_v_vec, tol);
    check_range_near(acc_rg_wel.variance_of_real_data(), re_v_vec, tol);

    check_near(acc_sv_std.variance_of_imag_data()[0], im_v, tol);
    check_range_near(acc_mva_std.variance_of_imag_data(), im_v_vec, tol);
    check_range_near(acc_rg_std.variance_of_imag_data(), im_v_vec, tol);
    check_near(acc_sv_wel.variance_of_imag_data()[0], im_v, tol);
    check_range_near(acc_mva_wel.variance_of_imag_data(), im_v_vec, tol);
    check_range_near(acc_rg_wel.variance_of_imag_data(), im_v_vec, tol);

    check_near(acc_sv_std.covariance_of_real_and_imag_data()[0], reim_v, tol);
    check_range_near(acc_mva_std.covariance_of_real_and_imag_data(), reim_v_vec, tol);
    check_range_near(acc_rg_std.covariance_of_real_and_imag_data(), reim_v_vec, tol);
    check_near(acc_sv_wel.covariance_of_real_and_imag_data()[0], reim_v, tol);
    check_range_near(acc_mva_wel.covariance_of_real_and_imag_data(), reim_v_vec, tol);
    check_range_near(acc_rg_wel.covariance_of_real_and_imag_data(), reim_v_vec, tol);

    // check reset and merging of accumulators
    acc_std merge_std(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    acc_wel merge_wel(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    ASSERT_EQ(merge_std.count(), 0);
    ASSERT_EQ(merge_wel.count(), 0);
    merge_std << acc_mva_std;
    merge_std << acc_rg_std;
    merge_wel << acc_mva_wel;
    merge_wel << acc_rg_wel;
    ASSERT_EQ(merge_std.count(), 2 * tp_c.total_count);
    ASSERT_EQ(merge_wel.count(), 2 * tp_c.total_count);
    check_range_near(merge_std.mean(), m_vec_c, tol);
    check_range_near(merge_std.covariance_of_real_and_imag_data(), reim_v_vec, tol);
    check_range_near(merge_wel.mean(), m_vec_c, tol);
    check_range_near(merge_wel.covariance_of_real_and_imag_data(), reim_v_vec, tol);
}