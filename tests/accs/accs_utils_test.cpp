#include "./accs_fixture.hpp"

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/ranges.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;

} // namespace

// Test sample type related concepts.
TEST_F(SimplemcAccs, UtilsSampleTypeConcepts) {
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

// Test mean utility function.
TEST_F(SimplemcAccs, UtilsMeanStandard) {
    // for standard algorithm, mdata = \sum_i x_i
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    for (const auto& v : vec_d_data) {
        mdata += v;
    }

    const auto result = accs::mean<standard>(mdata, vec_d_n);
    check_range_near(result, vec_d_mean, 1e-14);
}

TEST_F(SimplemcAccs, UtilsMeanWelford) {
    // for welford algorithm, mdata = running mean
    const auto result = accs::mean<welford>(vec_d_mean, vec_d_n);
    check_range_near(result, vec_d_mean, 1e-14);
}

TEST_F(SimplemcAccs, UtilsMeanEmpty) {
    // n=0 should return NaN
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    const auto result = accs::mean<standard>(mdata, 0);
    EXPECT_TRUE(simplemc::ranges::all_of(result, [](double x) { return std::isnan(x); }));
}

// Test diagonal covariance utility function.
TEST_F(SimplemcAccs, UtilsDiagCovarianceStandard) {
    // for standard algorithm, mdata = \sum_i x_i, cdata = \sum_i x_i * x_i
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Vector3d cdata = Eigen::Vector3d::Zero();
    for (const auto& v : vec_d_data) {
        mdata += v;
        cdata += v.cwiseProduct(v);
    }

    const auto result = accs::diag_covariance<standard>(mdata, mdata, cdata, vec_d_n);
    check_range_near(result, vec_d_var, 1e-14);
}

TEST_F(SimplemcAccs, UtilsDiagCovarianceWelford) {
    // for welford algorithm, mdata = running mean,
    // cdata = \sum_i (x_i - mean_i) * (x_{i-1} - mean_i)
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Vector3d cdata = Eigen::Vector3d::Zero();
    for (int i = 0; i < vec_d_n; ++i) {
        const Eigen::Vector3d tmp = vec_d_data[i] - mdata;
        mdata += tmp / static_cast<double>(i + 1);
        cdata += tmp.cwiseProduct(vec_d_data[i] - mdata);
    }

    const auto result = accs::diag_covariance<welford>(mdata, mdata, cdata, vec_d_n);
    check_range_near(result, vec_d_var, 1e-14);
}

TEST_F(SimplemcAccs, UtilsDiagCovarianceEdgeCases) {
    // n=0 or n=1 should return NaN
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    auto r0 = accs::diag_covariance<standard>(z, z, z, 0);
    auto r1 = accs::diag_covariance<standard>(z, z, z, 1);
    EXPECT_TRUE(simplemc::ranges::all_of(r0, [](double x) { return std::isnan(x); }));
    EXPECT_TRUE(simplemc::ranges::all_of(r1, [](double x) { return std::isnan(x); }));
}

// Test full covariance utility function.
TEST_F(SimplemcAccs, UtilsCovarianceStandard) {
    // for standard algorithm, mdata = \sum_i x_i, cdata = \sum_i x_i * x_i^T
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cdata = Eigen::Matrix3d::Zero();
    for (const auto& v : vec_d_data) {
        mdata += v;
        cdata += v * v.transpose();
    }

    const auto result = accs::covariance<standard>(mdata, mdata, cdata, vec_d_n);
    check_range_near(make_span(result), make_span(vec_d_cov), 1e-14);
}

TEST_F(SimplemcAccs, UtilsCovarianceWelford) {
    // for Welford algorithm, mdata = running mean,
    // cdata = \sum_i (x_i - mean_i) * (x_{i-1} - mean_i)^T
    Eigen::Vector3d mdata = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cdata = Eigen::Matrix3d::Zero();
    for (int i = 0; i < vec_d_n; ++i) {
        const Eigen::Vector3d tmp = vec_d_data[i] - mdata;
        mdata += tmp / static_cast<double>(i + 1);
        cdata += tmp * (vec_d_data[i] - mdata).transpose();
    }

    const auto result = accs::covariance<welford>(mdata, mdata, cdata, vec_d_n);
    check_range_near(make_span(result), make_span(vec_d_cov), 1e-14);
}

TEST_F(SimplemcAccs, UtilsCovarianceEdgeCases) {
    // n=0 or n=1 should return NaN matrix
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    Eigen::Matrix3d mz = Eigen::Matrix3d::Zero();
    auto r0 = accs::covariance<standard>(z, z, mz, 0);
    auto r1 = accs::covariance<standard>(z, z, mz, 1);
    EXPECT_TRUE(simplemc::ranges::all_of(make_span(r0), [](double x) { return std::isnan(x); }));
    EXPECT_TRUE(simplemc::ranges::all_of(make_span(r1), [](double x) { return std::isnan(x); }));
}

// Test tau (autocorrelation time) utility function.
TEST_F(SimplemcAccs, UtilsTauScalar) {
    // tau = 0.5 * (s_blocked * b / s_naive - 1)
    // if s_naive == s_blocked and b == 1, tau should be 0
    EXPECT_DOUBLE_EQ(accs::tau(1.0, 1.0, 1), 0.0);

    // s_blocked = 2.0, s_naive = 1.0, b = 2 =>
    // tau = 0.5 * (2 * 2 / 1 - 1) = 0.5 * 3 = 1.5
    EXPECT_DOUBLE_EQ(accs::tau(1.0, 2.0, 2), 1.5);

    // s_blocked = 3.0, s_naive = 2.0, b = 4 =>
    // tau = 0.5 * (3 * 4 / 2 - 1) = 0.5 * 5 = 2.5
    EXPECT_DOUBLE_EQ(accs::tau(2.0, 3.0, 4), 2.5);
}

TEST_F(SimplemcAccs, UtilsTauMatrix) {
    // 2x2 matrix version.
    Eigen::Matrix2d s_naive;
    s_naive << 1.0, 2.0, 2.0, 4.0;
    Eigen::Matrix2d s_blocked;
    s_blocked << 2.0, 4.0, 4.0, 8.0;
    std::uint64_t b = 3;

    // tau(i,j) = 0.5 * (s_blocked(i,j) * 3 / s_naive(i,j) - 1)
    // = 0.5 * (2 * 3 - 1) = 2.5 for all elements (since ratio is 2 everywhere).
    Eigen::Matrix2d expected = Eigen::Matrix2d::Constant(2.5);
    auto result = accs::tau(s_naive, s_blocked, b);
    check_range_near(make_span(result), make_span(expected), 1e-14);
}

// Test other utility functions.
TEST_F(SimplemcAccs, UtilsRandomSampleSize) {
    using namespace simplemc::detail;

    // scalar types have size 1
    EXPECT_EQ(random_sample_size(1.0), 1);
    EXPECT_EQ(random_sample_size(std::complex<double>(1.0)), 1);

    // vector types report their actual size
    Eigen::Vector3d v3 = Eigen::Vector3d::Zero();
    EXPECT_EQ(random_sample_size(v3), 3);

    Eigen::VectorXd vx(7);
    EXPECT_EQ(random_sample_size(vx), 7);
}

TEST_F(SimplemcAccs, UtilsZeroSample) {
    using namespace simplemc::detail;

    // scalar zero
    EXPECT_EQ(zero_sample(1.0), 0.0);
    EXPECT_EQ(zero_sample(std::complex<double>(1.0)), std::complex<double>(0.0));

    // vector zero
    Eigen::Vector3d v3 = Eigen::Vector3d::Ones();
    auto z3 = zero_sample(v3);
    EXPECT_EQ(z3.size(), 3);
    check_range_equal(z3, Eigen::Vector3d::Zero());
}
