/**
 * @file
 * @brief Unit tests for simplemc-accs.
 */

#include "./stochastic_process.hpp"
#include "../test_utils.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/utils/fmt_complex.hpp>

#include <fmt/ranges.h>
#include <range/v3/view/zip.hpp>

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

        auto s1_d = state_d { 1.0, 0.0, 0.0 };
        auto s2_d = state_d { 0.0, 1.0, 0.0 };
        auto s3_d = state_d { 0.0, 0.0, 1.0 };
        auto probs_d = std::vector<double> { 1.0, 1.0, 1.0 };
        Eigen::MatrixXd mcmc_props_d(3, 3);
        mcmc_props_d << 2, 3, 6, 1, 7, 4, 9, 9, 10;
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
        mcmc_props_d << 10, 3, 9, 8, 1, 2, 4, 4, 7;
        sp_c.set_states(std::vector<state_c> { s1_c, s2_c, s3_c });
        sp_c.set_weights(probs_c);
        sp_c.set_mcmc_proposal(mcmc_props_c);
        sp_c.mcmc_warmup(steps);
        sp_c.mcmc_run(steps);
    }

    proc_d sp_d;
    proc_c sp_c;
    int steps { 100000 };
};

// Fill accumulators with random samples.
template <typename A, typename T>
void fill_accs(A& acc_sv, A& acc_mva, A& acc_rg, const std::vector<T>& samples) {
    const auto size = acc_mva.size();
    const auto idxs = std::vector<long> { 0, 1, 2 };
    for (std::size_t i = 0; i < samples.size(); ++i) {
        auto mva = acc_mva.make_mva();
        acc_sv << samples[i](0);
        for (int j = 0; j < size; ++j) {
            mva[j] << samples[i](j);
        }
        mva.increment_count();
        if constexpr (requires (A& a) { a.check_and_add_block(); }) {
            acc_mva.check_and_add_block();
        }
        acc_rg.accumulate(samples[i], idxs);
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
    using acc_c = simplemc::mean_acc<std::complex<double>, simplemc::varalg::welford>;
    using storage_d = typename acc_d::vec_type;
    using storage_c = typename acc_c::vec_type;
    double tol = 1e-10;
    double tol_m = 1e-10;
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

    // check count of filled accumulators
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

    // check merging of accumulators
    constexpr auto merge_size = 20;
    acc_d merge_exp_d(storage_d::Constant(size, -12.7).eval());
    acc_d merge_1_d = merge_exp_d;
    acc_d merge_2_d(storage_d::Constant(size, 0.5).eval());
    acc_c merge_exp_c(storage_c::Constant(size, { 2.0, 1.0 }).eval());
    auto merge_1_c = merge_exp_c;
    acc_c merge_2_c(storage_c::Constant(size, { -1.7, 2.5 }).eval());
    auto merge_samples_d =
        std::vector<decltype(sp_d)::state_type>(sp_d.samples.begin(), sp_d.samples.begin() + merge_size);
    auto merge_samples_c =
        std::vector<decltype(sp_c)::state_type>(sp_c.samples.begin(), sp_c.samples.begin() + merge_size);
    for (int i = 0; i < merge_size; ++i) {
        merge_exp_d.accumulate(merge_samples_d[i]);
        merge_exp_c.accumulate(merge_samples_c[i]);
        if (i < merge_size / 2) {
            merge_1_d.accumulate(merge_samples_d[i]);
            merge_1_c.accumulate(merge_samples_c[i]);
        } else {
            merge_2_d.accumulate(merge_samples_d[i]);
            merge_2_c.accumulate(merge_samples_c[i]);
        }
    }
    merge_1_d << merge_2_d;
    merge_1_c << merge_2_c;
    ASSERT_EQ(merge_1_d.count(), merge_size);
    ASSERT_EQ(merge_1_c.count(), merge_size);
    check_range_near(merge_1_d.mdata(), merge_exp_d.mdata(), tol_m);
    check_range_near(merge_1_c.mdata(), merge_exp_c.mdata(), tol_m);
}

// Test variance accumulator for double values.
TEST_F(SimplemcAccs, DoubleVarianceAccumulator) {
    // general set up
    using acc_std = simplemc::var_acc<double, simplemc::varalg::standard>;
    using acc_wel = simplemc::var_acc<double, simplemc::varalg::welford>;
    using storage_d = typename acc_std::vec_type;
    double tol = 1e-10;
    double tol_m = 1e-10;
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

    // check count of filled accumulators
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

    // check merging of accumulators
    constexpr auto merge_size = 20;
    acc_std merge_exp_std(storage_d::Constant(size, -12.7).eval());
    acc_std merge_1_std = merge_exp_std;
    acc_std merge_2_std(storage_d::Constant(size, 0.5).eval());
    acc_std merge_exp_wel(storage_d::Constant(size, -12.7).eval());
    acc_std merge_1_wel = merge_exp_wel;
    acc_std merge_2_wel(storage_d::Constant(size, 0.5).eval());
    auto merge_samples =
        std::vector<decltype(sp_d)::state_type>(sp_d.samples.begin(), sp_d.samples.begin() + merge_size);
    for (int i = 0; i < merge_size; ++i) {
        merge_exp_std.accumulate(merge_samples[i]);
        merge_exp_wel.accumulate(merge_samples[i]);
        if (i < merge_size / 2) {
            merge_1_std.accumulate(merge_samples[i]);
            merge_1_wel.accumulate(merge_samples[i]);
        } else {
            merge_2_std.accumulate(merge_samples[i]);
            merge_2_wel.accumulate(merge_samples[i]);
        }
    }
    merge_1_std << merge_2_std;
    merge_1_wel << merge_2_wel;
    ASSERT_EQ(merge_1_std.count(), merge_exp_std.count());
    ASSERT_EQ(merge_1_wel.count(), merge_exp_wel.count());
    check_range_near(merge_1_std.mdata(), merge_exp_std.mdata(), tol_m);
    check_range_near(merge_1_std.vdata(), merge_exp_std.vdata(), tol_m);
    check_range_near(merge_1_wel.mdata(), merge_exp_wel.mdata(), tol_m);
    check_range_near(merge_1_wel.vdata(), merge_exp_wel.vdata(), tol_m);
}

// Test variance accumulator for complex values.
TEST_F(SimplemcAccs, ComplexVarianceAccumulator) {
    // general set up
    using acc_std = simplemc::var_acc<std::complex<double>, simplemc::varalg::standard>;
    using acc_wel = simplemc::var_acc<std::complex<double>, simplemc::varalg::welford>;
    using storage_c = typename acc_std::cplx_vec_type;
    double tol = 1e-10;
    double tol_m = 1e-10;
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

    // check count of filled accumulators
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

    // check merging of accumulators
    constexpr auto merge_size = 20;
    acc_std merge_exp_std(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    acc_std merge_1_std = merge_exp_std;
    acc_std merge_2_std(storage_c::Constant(size, { 0.3, 2.8 }).eval());
    acc_std merge_exp_wel(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    acc_std merge_1_wel = merge_exp_wel;
    acc_std merge_2_wel(storage_c::Constant(size, { 0.3, 2.8 }).eval());
    auto merge_samples =
        std::vector<decltype(sp_c)::state_type>(sp_c.samples.begin(), sp_c.samples.begin() + merge_size);
    for (int i = 0; i < merge_size; ++i) {
        merge_exp_std.accumulate(merge_samples[i]);
        merge_exp_wel.accumulate(merge_samples[i]);
        if (i < merge_size / 2) {
            merge_1_std.accumulate(merge_samples[i]);
            merge_1_wel.accumulate(merge_samples[i]);
        } else {
            merge_2_std.accumulate(merge_samples[i]);
            merge_2_wel.accumulate(merge_samples[i]);
        }
    }
    merge_1_std << merge_2_std;
    merge_1_wel << merge_2_wel;
    ASSERT_EQ(merge_1_std.count(), merge_exp_std.count());
    ASSERT_EQ(merge_1_wel.count(), merge_exp_wel.count());
    check_range_near(merge_1_std.mdata(), merge_exp_std.mdata(), tol_m);
    check_range_near(merge_1_std.rdata(), merge_exp_std.rdata(), tol_m);
    check_range_near(merge_1_std.idata(), merge_exp_std.idata(), tol_m);
    check_range_near(merge_1_std.cdata(), merge_exp_std.cdata(), tol_m);
    check_range_near(merge_1_wel.mdata(), merge_exp_wel.mdata(), tol_m);
    check_range_near(merge_1_wel.rdata(), merge_exp_wel.rdata(), tol_m);
    check_range_near(merge_1_wel.idata(), merge_exp_wel.idata(), tol_m);
    check_range_near(merge_1_wel.cdata(), merge_exp_wel.cdata(), tol_m);
}

// Test covariance accumulator for double values.
TEST_F(SimplemcAccs, DoubleCovarianceAccumulator) {
    // general set up
    using acc_std = simplemc::covar_acc<double, simplemc::varalg::standard>;
    using acc_wel = simplemc::covar_acc<double, simplemc::varalg::welford>;
    using storage_d = typename acc_std::vec_type;
    double tol = 1e-10;
    double tol_m = 1e-10;
    acc_std acc_sv_std(1, 5.0), acc_rg_std(size, 1.0);
    acc_wel acc_sv_wel(1, 5.0), acc_rg_wel(size, 1.0);

    // check size of empty accumulators
    ASSERT_EQ(acc_sv_std.size(), 1);
    ASSERT_EQ(acc_rg_std.size(), size);
    ASSERT_EQ(acc_sv_wel.size(), 1);
    ASSERT_EQ(acc_rg_wel.size(), size);

    // check empty accumulators
    check_empty(acc_sv_std);
    check_empty(acc_rg_std);
    check_empty(acc_sv_wel);
    check_empty(acc_rg_wel);

    // fill accumulators
    const auto idxs = std::vector<long> { 0, 1, 2 };
    for (auto& s : sp_d.samples) {
        acc_sv_std << s(0);
        acc_sv_wel << s(0);
        acc_rg_std.accumulate(s, idxs);
        acc_rg_wel.accumulate(s, idxs);
    }

    // check count of filled accumulators
    ASSERT_EQ(acc_sv_std.count(), sp_d.total_count);
    ASSERT_EQ(acc_rg_std.count(), sp_d.total_count);
    ASSERT_EQ(acc_sv_wel.count(), sp_d.total_count);
    ASSERT_EQ(acc_rg_wel.count(), sp_d.total_count);

    // check mean
    const auto m_d = sample_mean(sp_d);
    check_near(acc_sv_std.mean()[0], m_d[0], tol);
    check_range_near(acc_rg_std.mean(), m_d, tol);
    check_near(acc_sv_wel.mean()[0], m_d[0], tol);
    check_range_near(acc_rg_wel.mean(), m_d, tol);

    // check covariance
    const auto c_d = sample_covariance(sp_d);
    check_near(acc_sv_std.covariance_of_data()(0, 0), c_d(0, 0), tol);
    check_range_near(simplemc::make_span(acc_rg_std.covariance_of_data()), simplemc::make_span(c_d), tol);
    check_near(acc_sv_wel.covariance_of_data()(0, 0), c_d(0, 0), tol);
    check_range_near(simplemc::make_span(acc_rg_wel.covariance_of_data()), simplemc::make_span(c_d), tol);

    // check merging of accumulators
    constexpr auto merge_size = 20;
    acc_std merge_exp_std(storage_d::Constant(size, -12.7).eval());
    acc_std merge_1_std = merge_exp_std;
    acc_std merge_2_std(storage_d::Constant(size, 0.5).eval());
    acc_std merge_exp_wel(storage_d::Constant(size, -12.7).eval());
    acc_std merge_1_wel = merge_exp_wel;
    acc_std merge_2_wel(storage_d::Constant(size, 0.5).eval());
    auto merge_samples =
        std::vector<decltype(sp_d)::state_type>(sp_d.samples.begin(), sp_d.samples.begin() + merge_size);
    for (int i = 0; i < merge_size; ++i) {
        merge_exp_std.accumulate(merge_samples[i]);
        merge_exp_wel.accumulate(merge_samples[i]);
        if (i < merge_size / 2) {
            merge_1_std.accumulate(merge_samples[i]);
            merge_1_wel.accumulate(merge_samples[i]);
        } else {
            merge_2_std.accumulate(merge_samples[i]);
            merge_2_wel.accumulate(merge_samples[i]);
        }
    }
    merge_1_std << merge_2_std;
    merge_1_wel << merge_2_wel;
    ASSERT_EQ(merge_1_std.count(), merge_exp_std.count());
    ASSERT_EQ(merge_1_wel.count(), merge_exp_wel.count());
    check_range_near(merge_1_std.mdata(), merge_exp_std.mdata(), tol_m);
    check_range_near(simplemc::make_span(merge_1_std.cdata()), simplemc::make_span(merge_exp_std.cdata()), tol_m);
    check_range_near(merge_1_wel.mdata(), merge_exp_wel.mdata(), tol_m);
    check_range_near(simplemc::make_span(merge_1_wel.cdata()), simplemc::make_span(merge_exp_wel.cdata()), tol_m);
}

// Test covariance accumulator for complex values.
TEST_F(SimplemcAccs, ComplexCovarianceAccumulator) {
    // general set up
    using acc_std = simplemc::covar_acc<std::complex<double>, simplemc::varalg::standard>;
    using acc_wel = simplemc::covar_acc<std::complex<double>, simplemc::varalg::welford>;
    using storage_c = typename acc_std::cplx_vec_type;
    double tol = 1e-10;
    double tol_m = 1e-10;
    acc_std acc_sv_std(1, { 5.0, 1.0 }), acc_rg_std(size, 10.0);
    acc_wel acc_sv_wel(1, { 5.0, 1.0 }), acc_rg_wel(size, 10.0);

    // check size of empty accumulators
    ASSERT_EQ(acc_sv_std.size(), 1);
    ASSERT_EQ(acc_rg_std.size(), size);
    ASSERT_EQ(acc_sv_wel.size(), 1);
    ASSERT_EQ(acc_rg_wel.size(), size);

    // check empty accumulators
    check_empty(acc_sv_std);
    check_empty(acc_rg_std);
    check_empty(acc_sv_wel);
    check_empty(acc_rg_wel);

    // fill accumulators
    const auto idxs = std::vector<long> { 0, 1, 2 };
    for (auto& s : sp_c.samples) {
        acc_sv_std << s(0);
        acc_sv_wel << s(0);
        acc_rg_std.accumulate(s, idxs);
        acc_rg_wel.accumulate(s, idxs);
    }

    // check count of filled accumulators
    ASSERT_EQ(acc_sv_std.count(), sp_c.total_count);
    ASSERT_EQ(acc_rg_std.count(), sp_c.total_count);
    ASSERT_EQ(acc_sv_wel.count(), sp_c.total_count);
    ASSERT_EQ(acc_rg_wel.count(), sp_c.total_count);

    // check mean
    const auto m_c = sample_mean(sp_c);
    check_near(acc_sv_std.mean()[0], m_c[0], tol);
    check_range_near(acc_rg_std.mean(), m_c, tol);
    check_near(acc_sv_wel.mean()[0], m_c[0], tol);
    check_range_near(acc_rg_wel.mean(), m_c, tol);

    // check covariance
    const auto [re_c, im_c, reim_c] = sample_covariance(sp_c);

    check_near(acc_sv_std.covariance_of_real_data()(0, 0), re_c(0, 0), tol);
    check_range_near(simplemc::make_span(acc_rg_std.covariance_of_real_data()), simplemc::make_span(re_c), tol);
    check_near(acc_sv_wel.covariance_of_real_data()(0, 0), re_c(0, 0), tol);
    check_range_near(simplemc::make_span(acc_rg_wel.covariance_of_real_data()), simplemc::make_span(re_c), tol);

    check_near(acc_sv_std.covariance_of_imag_data()(0, 0), im_c(0, 0), tol);
    check_range_near(simplemc::make_span(acc_rg_std.covariance_of_imag_data()), simplemc::make_span(im_c), tol);
    check_near(acc_sv_wel.covariance_of_imag_data()(0, 0), im_c(0, 0), tol);
    check_range_near(simplemc::make_span(acc_rg_wel.covariance_of_imag_data()), simplemc::make_span(im_c), tol);

    check_near(acc_sv_std.covariance_of_real_and_imag_data()(0, 0), reim_c(0, 0), tol);
    check_range_near(
        simplemc::make_span(acc_rg_std.covariance_of_real_and_imag_data()), simplemc::make_span(reim_c), tol);
    check_near(acc_sv_wel.covariance_of_real_and_imag_data()(0, 0), reim_c(0, 0), tol);
    check_range_near(
        simplemc::make_span(acc_rg_wel.covariance_of_real_and_imag_data()), simplemc::make_span(reim_c), tol);

    // check merging of accumulators
    constexpr auto merge_size = 20;
    acc_std merge_exp_std(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    acc_std merge_1_std = merge_exp_std;
    acc_std merge_2_std(storage_c::Constant(size, { 0.3, 2.8 }).eval());
    acc_std merge_exp_wel(storage_c::Constant(size, { -12.7, 4.2 }).eval());
    acc_std merge_1_wel = merge_exp_wel;
    acc_std merge_2_wel(storage_c::Constant(size, { 0.3, 2.8 }).eval());
    auto merge_samples =
        std::vector<decltype(sp_c)::state_type>(sp_c.samples.begin(), sp_c.samples.begin() + merge_size);
    for (int i = 0; i < merge_size; ++i) {
        merge_exp_std.accumulate(merge_samples[i]);
        merge_exp_wel.accumulate(merge_samples[i]);
        if (i < merge_size / 2) {
            merge_1_std.accumulate(merge_samples[i]);
            merge_1_wel.accumulate(merge_samples[i]);
        } else {
            merge_2_std.accumulate(merge_samples[i]);
            merge_2_wel.accumulate(merge_samples[i]);
        }
    }
    merge_1_std << merge_2_std;
    merge_1_wel << merge_2_wel;
    ASSERT_EQ(merge_1_std.count(), merge_exp_std.count());
    ASSERT_EQ(merge_1_wel.count(), merge_exp_wel.count());
    check_range_near(merge_1_std.mdata(), merge_exp_std.mdata(), tol_m);
    check_range_near(simplemc::make_span(merge_1_std.rdata()), simplemc::make_span(merge_exp_std.rdata()), tol_m);
    check_range_near(simplemc::make_span(merge_1_std.idata()), simplemc::make_span(merge_exp_std.idata()), tol_m);
    check_range_near(simplemc::make_span(merge_1_std.cdata()), simplemc::make_span(merge_exp_std.cdata()), tol_m);
    check_range_near(merge_1_wel.mdata(), merge_exp_wel.mdata(), tol_m);
    check_range_near(simplemc::make_span(merge_1_wel.rdata()), simplemc::make_span(merge_exp_wel.rdata()), tol_m);
    check_range_near(simplemc::make_span(merge_1_wel.idata()), simplemc::make_span(merge_exp_wel.idata()), tol_m);
    check_range_near(simplemc::make_span(merge_1_wel.cdata()), simplemc::make_span(merge_exp_wel.cdata()), tol_m);
}

// Test block accumulator.
TEST_F(SimplemcAccs, BlockAccumulator) {
    // general set up
    using acc_d = simplemc::block_acc<simplemc::var_acc<double>>;
    double tol = 1e-10;
    int bs_sv = 1;
    int bs_mva = 100;
    int bs_rg = 5000;
    acc_d acc_sv_d(1, 5.0), acc_mva_d(size, 2.0, bs_mva), acc_rg_d(size, 1.0, bs_rg);

    // fill accumulators
    fill_accs(acc_sv_d, acc_mva_d, acc_rg_d, sp_d.samples);

    // check count of filled accumulators
    ASSERT_EQ(acc_sv_d.count(), sp_d.total_count / bs_sv);
    ASSERT_EQ(acc_mva_d.count(), sp_d.total_count / bs_mva);
    ASSERT_EQ(acc_rg_d.count(), sp_d.total_count / bs_rg);

    // check mean
    const auto bsp_sv = block_samples(sp_d, bs_sv);
    const auto bsp_mva = block_samples(sp_d, bs_mva);
    const auto bsp_rg = block_samples(sp_d, bs_rg);
    check_near(acc_sv_d.accumulator().mean()[0], sample_mean(bsp_sv)[0], tol);
    check_range_near(acc_mva_d.accumulator().mean(), sample_mean(bsp_mva), tol);
    check_range_near(acc_rg_d.accumulator().mean(), sample_mean(bsp_rg), tol);

    // check stderror/variance
    check_near(acc_sv_d.accumulator().variance_of_data()[0], sample_variance(bsp_sv)[0], tol);
    check_range_near(acc_mva_d.accumulator().variance_of_data(), sample_variance(bsp_mva), tol);
    check_range_near(acc_rg_d.accumulator().variance_of_data(), sample_variance(bsp_rg), tol);
}

// Test autocorrelation accumulator.
TEST_F(SimplemcAccs, AutocorrelationAccumulator) {
    // general set up
    using acc_std = simplemc::autocorr_acc<simplemc::var_acc<double, simplemc::varalg::standard>>;
    using acc_wel = simplemc::autocorr_acc<simplemc::var_acc<double, simplemc::varalg::welford>>;
    double tol = 1e-10;
    acc_std acc_sv_std(1, 5.0), acc_rg_std(size, 1.0);
    acc_wel acc_sv_wel(1, 5.0), acc_rg_wel(size, 1.0);

    // check size of empty accumulators
    ASSERT_EQ(acc_sv_std.size(), 1);
    ASSERT_EQ(acc_rg_std.size(), size);
    ASSERT_EQ(acc_sv_wel.size(), 1);
    ASSERT_EQ(acc_rg_wel.size(), size);

    // check empty accumulators
    check_empty(acc_sv_std);
    check_empty(acc_rg_std);
    check_empty(acc_sv_wel);
    check_empty(acc_rg_wel);

    // fill accumulators
    const auto idxs = std::vector<long> { 0, 1, 2 };
    for (auto& s : sp_d.samples) {
        acc_sv_std << s(0);
        acc_sv_wel << s(0);
        acc_rg_std.accumulate(s, idxs);
        acc_rg_wel.accumulate(s, idxs);
    }

    // check block variance accumulators with increasing block sizes and autocorrelation times
    auto [bsp_d, btau_d] = blocking_autocorr(sp_d);
    const auto max_level = acc_sv_std.find_level(2);
    const auto v0 = acc_rg_wel.accumulators()[0].variance_of_data();
    for (std::size_t i = 0; i < max_level; ++i) {
        const auto v_d = sample_variance(bsp_d[i]);
        check_near(acc_sv_std.accumulators()[i].variance_of_data()[0], v_d[0]);
        check_near(acc_sv_wel.accumulators()[i].variance_of_data()[0], v_d[0]);
        check_range_near(acc_rg_std.accumulators()[i].variance_of_data(), v_d);
        check_range_near(acc_rg_wel.accumulators()[i].variance_of_data(), v_d);
        const auto vi = acc_rg_wel.accumulators()[i].variance_of_data();
        const auto tau = acc_rg_wel.tau(v0, vi, acc_rg_wel.block_size(i));
        check_range_near(tau, btau_d[i], tol);
    }
}
