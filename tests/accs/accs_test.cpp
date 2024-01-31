/**
 * @file accs_test.cpp
 * @brief Unit tests for simplemc-accs.
 */

#include "../test_utils.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <random>

namespace {

constexpr std::uint32_t default_seed = 0x8a34e234;
constexpr double exp_mean = 2.0;
constexpr double exp_err = 1.0;

} // namespace

// Fill accumulators with random samples from a normal distribution.
template <typename A>
void fill_accs(A& acc_sv, A& acc_mva, A& acc_rg, int num) {
    using value_type = typename A::value_type;
    std::mt19937 eng { default_seed };
    std::normal_distribution<double> norm_dist { exp_mean, exp_err };
    const auto size = acc_mva.size();
    std::vector<value_type> vec(size);
    for (int i = 0; i < num; ++i) {
        auto mva = acc_mva.make_mva();
        if constexpr (std::is_same_v<typename A::value_type, double>) {
            acc_sv << norm_dist(eng);
            for (int j = 0; j < size; ++j) {
                mva[j] << norm_dist(eng); 
                vec[j] = norm_dist(eng);
            }
            acc_rg.accumulate(vec);
        } else {
            acc_sv << std::complex<double> { norm_dist(eng), norm_dist(eng) };
            for (int j = 0; j < size; ++j) {
                mva[j] << std::complex<double> { norm_dist(eng), norm_dist(eng) };
                vec[j] = std::complex<double> { norm_dist(eng), norm_dist(eng) };
            }
            acc_rg.accumulate(vec);
        }
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

// Check mean of accumulator.
template <typename A>
void check_mean(const A& acc, int n, double tol) {
    auto check = [tol](auto val) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(val)>, double>) {
            ASSERT_NEAR(val, exp_mean, tol);
        } else {
            ASSERT_NEAR(val.real(), exp_mean, tol);
            ASSERT_NEAR(val.imag(), exp_mean, tol);
        }
    };
    ASSERT_EQ(acc.count(), n);
    auto mean = acc.mean();
    for (int i = 0; i < acc.size(); ++i) {
        check(mean[i]);
    }
}

// Test utility functions.
TEST(SimplemcAccs, Utils) {
    using value_type = std::complex<double>;
    using storage_type = Eigen::ArrayX<value_type>;
    storage_type data = storage_type::Constant(5, std::complex<double> { 1.0, 1.0 });
    for (auto i = 0; i < data.size(); ++i) {
        data[i] = std::complex<double> { i * 1.0, i * 1.0 };
    }
    auto mean = simplemc::accs::mean(data, 5);
    auto mean_nan = simplemc::accs::mean(data, 0);
    for (auto i = 0; i < mean.size(); ++i) {
        check_near(mean[i], std::complex<double> { i / 5.0, i / 5.0 });
        ASSERT_TRUE(std::isnan(mean_nan[i].real()));
        ASSERT_TRUE(std::isnan(mean_nan[i].imag()));
    }
}

// Test mean accumulator.
TEST(SimplemcAccs, MeanAccumulator) {
    // general set up
    using acc_d = simplemc::mean_acc<double>;
    using acc_c = simplemc::mean_acc<std::complex<double>, simplemc::accs::varalg::welford>; 
    int n = 100000;
    double tol = 1e-2;
    int size = 3;
    acc_d acc_sv(1, 5.0), acc_mva(size, 2.0), acc_rg(size, 2.0);
    acc_c acc_sv_w(1, { 5.0, 1.0 }), acc_mva_w(size, { -3.2, -2.2 }), acc_rg_w(size, 10.0);

    // check empty accumulators
    ASSERT_EQ(acc_sv.size(), 1);
    ASSERT_EQ(acc_mva.size(), size);
    ASSERT_EQ(acc_rg.size(), size);
    ASSERT_EQ(acc_sv_w.size(), 1);
    ASSERT_EQ(acc_mva_w.size(), size);
    ASSERT_EQ(acc_rg_w.size(), size);
    check_empty(acc_sv);
    check_empty(acc_mva);
    check_empty(acc_rg);
    check_empty(acc_sv_w);
    check_empty(acc_mva_w);
    check_empty(acc_rg_w);

    // fill accumulators
    fill_accs(acc_sv, acc_mva, acc_rg, n);
    fill_accs(acc_sv_w, acc_mva_w, acc_rg_w, n);

    // check filled accumulators
    ASSERT_EQ(acc_sv.count(), n);
    ASSERT_EQ(acc_mva.count(), n);
    ASSERT_EQ(acc_rg.count(), n);
    ASSERT_EQ(acc_sv_w.count(), n);
    ASSERT_EQ(acc_mva_w.count(), n);
    ASSERT_EQ(acc_rg_w.count(), n);
    check_mean(acc_sv, n, tol);
    check_mean(acc_mva, n, tol);
    check_mean(acc_rg, n, tol);
    check_mean(acc_sv_w, n, tol);
    check_mean(acc_mva_w, n, tol);
    check_mean(acc_rg_w, n, tol);

    // check reset and merging of accumulators
    acc_d merge(size);
    merge.reset(0, 2.0);
    ASSERT_EQ(merge.count(), 0);
    ASSERT_EQ(merge.shift(), 2.0);
    merge << acc_mva;
    merge << acc_rg;
    ASSERT_EQ(merge.count(), 2 * n);
    check_mean(merge, 2 * n, tol);
}