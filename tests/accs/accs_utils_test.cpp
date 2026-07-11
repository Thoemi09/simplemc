// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "./accs_test_traits.hpp"

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/ranges.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace {

using namespace simplemc;
using namespace test_detail;
constexpr varalg standard = varalg::standard;
constexpr varalg welford = varalg::welford;

} // namespace

// Concept tests.
TEST(SimplemcAccsUtils, SampleTypeConcepts) {
    // sample_scalar
    static_assert(sample_scalar<double>);
    static_assert(sample_scalar<std::complex<double>>);
    static_assert(!sample_scalar<int>);
    static_assert(!sample_scalar<float>);

    // sample_vector
    static_assert(sample_vector<Eigen::Vector3d>);
    static_assert(sample_vector<Eigen::VectorXd>);
    static_assert(sample_vector<Eigen::Vector2cd>);
    static_assert(sample_vector<Eigen::VectorXcd>);
    static_assert(!sample_vector<double>);

    // sample_type
    static_assert(sample_type<double>);
    static_assert(sample_type<std::complex<double>>);
    static_assert(sample_type<Eigen::Vector3d>);
    static_assert(sample_type<Eigen::VectorXd>);

    // sample_range
    static_assert(sample_range<std::vector<double>>);
    static_assert(sample_range<std::vector<Eigen::VectorXd>>);
    static_assert(!sample_range<std::vector<int>>);
}

// Mean utility function tests using the N=100 deterministic dataset.
TEST(SimplemcAccsUtils, MeanStandard) {
    auto data = make_data<Eigen::Vector3d>();
    // for standard algorithm, mdata = sum of all samples
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    for (const auto& v : data) {
        mdata += v;
    }

    const auto result = accs::mean<standard>(mdata, N);
    check_near(result, expected_mean<Eigen::Vector3d>(), 1e-10);
}

TEST(SimplemcAccsUtils, MeanWelford) {
    // for Welford algorithm, mdata = running mean
    const auto result = accs::mean<welford>(expected_mean<Eigen::Vector3d>(), N);
    check_near(result, expected_mean<Eigen::Vector3d>(), 1e-14);
}

TEST(SimplemcAccsUtils, MeanEmpty) {
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    const auto result = accs::mean<standard>(mdata, 0);
    EXPECT_TRUE(simplemc::ranges::all_of(result, [](double x) { return std::isnan(x); }));
}

// Diagonal covariance utility function tests using the N=100 deterministic dataset.
TEST(SimplemcAccsUtils, DiagCovarianceStandard) {
    auto data = make_data<Eigen::Vector3d>();
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Vector3d cdata = Eigen::Vector3d::Zero();
    for (const auto& v : data) {
        mdata += v;
        cdata += v.cwiseProduct(v);
    }

    const auto result = accs::diag_covariance<standard>(mdata, mdata, cdata, N);
    check_near(result, expected_var_data<Eigen::Vector3d>(), 1e-8);
}

TEST(SimplemcAccsUtils, DiagCovarianceWelford) {
    auto data = make_data<Eigen::Vector3d>();
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Vector3d cdata = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d tmp = data[i] - mdata;
        mdata += tmp / static_cast<double>(i + 1);
        cdata += tmp.cwiseProduct(data[i] - mdata);
    }

    const auto result = accs::diag_covariance<welford>(mdata, mdata, cdata, N);
    check_near(result, expected_var_data<Eigen::Vector3d>(), 1e-8);
}

TEST(SimplemcAccsUtils, DiagCovarianceEdgeCases) {
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    auto r0 = accs::diag_covariance<standard>(z, z, z, 0);
    auto r1 = accs::diag_covariance<standard>(z, z, z, 1);
    EXPECT_TRUE(simplemc::ranges::all_of(r0, [](double x) { return std::isnan(x); }));
    EXPECT_TRUE(simplemc::ranges::all_of(r1, [](double x) { return std::isnan(x); }));
}

// Full covariance utility function tests using the N=100 deterministic dataset.
TEST(SimplemcAccsUtils, CovarianceStandard) {
    auto data = make_data<Eigen::Vector3d>();
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cdata = Eigen::Matrix3d::Zero();
    for (const auto& v : data) {
        mdata += v;
        cdata += v * v.transpose();
    }

    const auto result = accs::covariance<standard>(mdata, mdata, cdata, N);
    check_near(make_span(result), make_span(expected_covar_data<Eigen::Vector3d>()), 1e-8);
}

TEST(SimplemcAccsUtils, CovarianceWelford) {
    auto data = make_data<Eigen::Vector3d>();
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cdata = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d tmp = data[i] - mdata;
        mdata += tmp / static_cast<double>(i + 1);
        cdata += tmp * (data[i] - mdata).transpose();
    }

    const auto result = accs::covariance<welford>(mdata, mdata, cdata, N);
    check_near(make_span(result), make_span(expected_covar_data<Eigen::Vector3d>()), 1e-8);
}

TEST(SimplemcAccsUtils, CovarianceEdgeCases) {
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    Eigen::Matrix3d mz = Eigen::Matrix3d::Zero();
    auto r0 = accs::covariance<standard>(z, z, mz, 0);
    auto r1 = accs::covariance<standard>(z, z, mz, 1);
    EXPECT_TRUE(simplemc::ranges::all_of(make_span(r0), [](double x) { return std::isnan(x); }));
    EXPECT_TRUE(simplemc::ranges::all_of(make_span(r1), [](double x) { return std::isnan(x); }));
}

// Tau (autocorrelation time) utility function tests.
TEST(SimplemcAccsUtils, TauScalar) {
    EXPECT_DOUBLE_EQ(accs::tau(1.0, 1.0, 1), 0.0);
    EXPECT_DOUBLE_EQ(accs::tau(1.0, 2.0, 2), 1.5);
    EXPECT_DOUBLE_EQ(accs::tau(2.0, 3.0, 4), 2.5);
}

TEST(SimplemcAccsUtils, TauMatrix) {
    Eigen::Matrix2d s_naive;
    s_naive << 1.0, 2.0, 2.0, 4.0;
    Eigen::Matrix2d s_blocked;
    s_blocked << 2.0, 4.0, 4.0, 8.0;
    std::uint64_t b = 3;

    Eigen::Matrix2d expected = Eigen::Matrix2d::Constant(2.5);
    auto result = accs::tau(s_naive, s_blocked, b);
    check_near(make_span(result), make_span(expected), 1e-14);
}

// Other utility function tests.
TEST(SimplemcAccsUtils, RandomSampleSize) {
    using namespace simplemc::detail;

    EXPECT_EQ(random_sample_size(1.0), 1);
    EXPECT_EQ(random_sample_size(std::complex<double>(1.0)), 1);

    Eigen::Vector3d v3 = Eigen::Vector3d::Zero();
    EXPECT_EQ(random_sample_size(v3), 3);

    Eigen::VectorXd vx(7);
    EXPECT_EQ(random_sample_size(vx), 7);
}

TEST(SimplemcAccsUtils, ZeroSample) {
    using namespace simplemc::detail;

    EXPECT_EQ(zero_sample(1.0), 0.0);
    EXPECT_EQ(zero_sample(std::complex<double>(1.0)), std::complex<double>(0.0));

    Eigen::Vector3d v3 = Eigen::Vector3d::Ones();
    auto z3 = zero_sample(v3);
    EXPECT_EQ(z3.size(), 3);
    check_equal(z3, Eigen::Vector3d::Zero());
}

// Standard error tests.
template <typename Tag>
class SimplemcAccsStderror : public ::testing::Test {};
TYPED_TEST_SUITE(SimplemcAccsStderror, AllAccTypes);

TYPED_TEST(SimplemcAccsStderror, StderrorEqualsSqrtVariance) {
    using T = typename TypeParam::sample_t;
    constexpr varalg A = TypeParam::alg;

    auto data = make_data<T>();
    auto acc = make_empty<var_acc<T, A>, T>();
    for (const auto& x : data) {
        acc << x;
    }

    auto se = stderror(acc);
    auto var = acc.variance();

    if constexpr (is_scalar_sample_v<T>) {
        EXPECT_DOUBLE_EQ(se, std::sqrt(var));
        EXPECT_GT(se, 0.0);
    } else {
        auto expected_se = var.cwiseSqrt().eval();
        check_near(se, expected_se, 1e-14);
        for (long j = 0; j < se.size(); ++j) {
            EXPECT_GT(se(j), 0.0);
        }
    }
}
