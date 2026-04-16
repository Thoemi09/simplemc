#include "./accs_test_traits.hpp"

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using namespace simplemc;
using namespace test_detail;

// All sample types and both varalg values for jackknife tests.
using JackknifeAccTypes = AllAccTypes;

} // namespace

// ================================================================================================
// jackknife_result concept checks
// ================================================================================================

template <typename Tag>
class SimplemcAccsJackknife : public ::testing::Test {};
TYPED_TEST_SUITE(SimplemcAccsJackknife, JackknifeAccTypes);

TYPED_TEST(SimplemcAccsJackknife, ResultSatisfiesConcepts) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    // jackknife_result wrapping var_acc has variance()
    using jk_var_t = jackknife_result<var_acc<T, A>>;
    static_assert(requires(const jk_var_t& r) { r.variance(); });
    static_assert(requires(const jk_var_t& r) { r.mean(); });
    static_assert(requires(const jk_var_t& r) { r.bias(); });
    static_assert(requires(const jk_var_t& r) { r.bias_corrected_mean(); });

    // jackknife_result wrapping covar_acc has covariance()
    using jk_covar_t = jackknife_result<covar_acc<T, A>>;
    static_assert(requires(const jk_covar_t& r) { r.covariance(); });
}

// ================================================================================================
// Identity jackknife (f = identity, single range)
// ================================================================================================

TYPED_TEST(SimplemcAccsJackknife, IdentityVariance) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    auto result = make_jackknife_var_acc<A>(data);

    // jackknife mean should equal the data mean
    ASSERT_EQ(result.count(), N);
    check_near(result.original_mean(), expected_mean<T>(), 1e-10);
    check_near(result.mean(), expected_mean<T>(), 1e-10);

    // For identity transformation, bias should be zero
    if constexpr (is_scalar_sample_v<T>) {
        check_near(result.bias(), T { 0 }, 1e-8);
    } else {
        for (long j = 0; j < result.size(); ++j) {
            check_near(result.bias()(j), typename T::Scalar { 0 }, 1e-8);
        }
    }

    // Jackknife variance of identity = (N-1)^2 * var_acc.variance()
    // For the identity transformation on uncorrelated data, the jackknife variance
    // should equal the variance of the mean: Var(X) / N.
    // For complex types, Var = Var(re) + Var(im).
    if constexpr (is_complex_sample_v<T>) {
        if constexpr (is_scalar_sample_v<T>) {
            auto expected_var_mean =
                (expected_var_real_data<T>() + expected_var_imag_data<T>()) / static_cast<double>(N);
            check_near(result.variance(), expected_var_mean, 1e-6);
        } else {
            real_vec_t<T> expected_var_mean =
                (expected_var_real_data<T>() + expected_var_imag_data<T>()) / static_cast<double>(N);
            check_near(result.variance(), expected_var_mean, 1e-6);
        }
    } else if constexpr (is_scalar_sample_v<T>) {
        auto expected_var_mean = expected_var_data<T>() / static_cast<double>(N);
        check_near(result.variance(), expected_var_mean, 1e-6);
    } else {
        real_vec_t<T> expected_var_mean = expected_var_data<T>() / static_cast<double>(N);
        check_near(result.variance(), expected_var_mean, 1e-6);
    }
}

TYPED_TEST(SimplemcAccsJackknife, IdentityCovariance) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    auto result = make_jackknife_covar_acc<A>(data);

    // jackknife mean should equal the data mean
    ASSERT_EQ(result.count(), N);
    check_near(result.original_mean(), expected_mean<T>(), 1e-10);
    check_near(result.mean(), expected_mean<T>(), 1e-10);

    // covariance should equal the covariance of the mean
    // For complex types, Cov = Cov(re,re) + Cov(im,im) + i*(Cov(re,im)^T - Cov(re,im)).
    // For this test data Cov(re,im) is symmetric, so the imaginary part vanishes.
    if constexpr (is_complex_sample_v<T>) {
        if constexpr (is_scalar_sample_v<T>) {
            auto expected_covar_mean =
                (expected_cov_rr<T>() + expected_cov_ii<T>()) / static_cast<double>(N);
            check_near(result.covariance(), expected_covar_mean, 1e-6);
        } else {
            // The full complex covariance: Cov(re,re) + Cov(im,im) + i*(Cov(re,im)^T - Cov(re,im)).
            // For this test data Cov(re,im) is symmetric, so the imaginary part vanishes.
            auto covar = result.covariance();
            real_mat_t<T> expected_real =
                (expected_cov_rr<T>() + expected_cov_ii<T>()) / static_cast<double>(N);
            real_mat_t<T> covar_re = covar.real();
            real_mat_t<T> covar_im = covar.imag();
            real_mat_t<T> zero = real_mat_t<T>::Zero(result.size(), result.size());
            check_near(covar_re, expected_real, 1e-6);
            check_near(covar_im, zero, 1e-6);
        }
    } else if constexpr (is_scalar_sample_v<T>) {
        auto expected_covar_mean = expected_covar_data<T>() / static_cast<double>(N);
        check_near(result.covariance(), expected_covar_mean, 1e-6);
    } else {
        real_mat_t<T> expected_covar_mean = expected_covar_data<T>() / static_cast<double>(N);
        check_near(result.covariance(), expected_covar_mean, 1e-6);
    }
}

// ================================================================================================
// Transformation function (single range)
// ================================================================================================

TEST(SimplemcAccsJackknifeSingle, SquareTransformation) {
    // f(x) = x^2. For x_i = i+1 (i = 0..99), mean(x) = 50.5.
    // Jackknife of f(x) = x^2 should estimate the variance of the sample mean of x^2.
    std::vector<double> data;
    data.reserve(N);
    for (int i = 0; i < N; ++i) {
        data.push_back(static_cast<double>(i + 1));
    }

    auto f_square = [](double x) { return x * x; };
    auto result = make_jackknife_var_acc(data, f_square);

    ASSERT_EQ(result.count(), N);

    // original_mean = f(mean(x)) = 50.5^2 = 2550.25
    ASSERT_NEAR(result.naive_mean(), 50.5 * 50.5, 1e-10);

    // For a nonlinear transformation, bias should be nonzero
    ASSERT_GT(std::abs(result.bias()), 0.0);

    // Variance should be positive
    ASSERT_GT(result.variance(), 0.0);

    // bias_corrected_mean = N * original_mean - (N-1) * mean
    double expected_bcm = N * result.naive_mean() - (N - 1.0) * result.mean();
    ASSERT_NEAR(result.bias_corrected_mean(), expected_bcm, 1e-8);
}

// ================================================================================================
// Multi-range jackknife (ratio estimation)
// ================================================================================================

TEST(SimplemcAccsJackknifeMulti, RatioEstimation) {
    // Two observables: x_i = i+1, y_i = 2*(i+1). f(x,y) = x/y.
    // True ratio = 0.5. Naive estimate f(mean_x, mean_y) = 50.5 / 101 = 0.5. No bias here
    // because the ratio of linear functions of the same variable simplifies.
    constexpr int n = 50;
    std::vector<double> xs, ys;
    xs.reserve(n);
    ys.reserve(n);
    for (int i = 0; i < n; ++i) {
        xs.push_back(static_cast<double>(i + 1));
        ys.push_back(2.0 * static_cast<double>(i + 1));
    }

    auto f_ratio = [](double x, double y) { return x / y; };
    auto result = make_jackknife_var_acc(f_ratio, xs, ys);

    ASSERT_EQ(result.count(), n);
    ASSERT_NEAR(result.naive_mean(), 0.5, 1e-14);
    ASSERT_NEAR(result.mean(), 0.5, 1e-10);

    // variance should be non-negative and small
    ASSERT_GE(result.variance(), 0.0);
    ASSERT_LT(result.variance(), 1e-2);
}

TEST(SimplemcAccsJackknifeMulti, RatioCovariance) {
    constexpr int n = 50;
    std::vector<double> xs, ys;
    xs.reserve(n);
    ys.reserve(n);
    for (int i = 0; i < n; ++i) {
        xs.push_back(static_cast<double>(i + 1));
        ys.push_back(2.0 * static_cast<double>(i + 1));
    }

    auto f_ratio = [](double x, double y) { return x / y; };
    auto result = make_jackknife_covar_acc(f_ratio, xs, ys);

    ASSERT_NEAR(result.naive_mean(), 0.5, 1e-14);
    // For scalar output, covariance() == variance()
    ASSERT_NEAR(result.covariance(), result.variance(), 1e-14);
}

// ================================================================================================
// batch_acc-based jackknife
// ================================================================================================

TYPED_TEST(SimplemcAccsJackknife, BatchAccIdentity) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();

    // Create batch_acc and accumulate data
    constexpr std::size_t m_b = 4;
    auto bacc = make_empty_batch<batch_acc<T, A>, T>(m_b);
    for (const auto& x : data) {
        bacc << x;
    }

    // Jackknife from batch_acc
    auto result = make_jackknife_var_acc(bacc);

    // Number of jackknife samples = number of full batches
    auto n_batches = bacc.num_full_batches();
    ASSERT_EQ(result.count(), static_cast<decltype(result.count())>(n_batches));
    ASSERT_GT(result.count(), 1U);

    // The batch jackknife variance should be non-negative
    if constexpr (is_scalar_sample_v<T>) {
        ASSERT_GE(result.variance(), 0.0);
    }

    // Compare with range-based jackknife on the same batch means
    auto bs = bacc.batches();
    std::vector<T> means;
    means.reserve(bs.size());
    for (const auto& b : bs) {
        means.push_back(b.mean());
    }
    auto result_range = make_jackknife_var_acc<A>(means);

    // original_mean should match the mean of the batch means (not bacc.mean()
    // which includes partial batches)
    check_near(result.original_mean(), result_range.original_mean(), 1e-10);
    check_near(result.mean(), result_range.mean(), 1e-10);
    check_near(result.variance(), result_range.variance(), 1e-10);
}

// ================================================================================================
// Multi batch_acc jackknife
// ================================================================================================

TEST(SimplemcAccsJackknifeBatch, MultiBatchRatio) {
    constexpr int n = 200;
    constexpr std::size_t m_b = 4;

    batch_acc<double> bacc_x(1, m_b);
    batch_acc<double> bacc_y(1, m_b);

    for (int i = 0; i < n; ++i) {
        bacc_x << static_cast<double>(i + 1);
        bacc_y << 2.0 * static_cast<double>(i + 1);
    }

    auto f_ratio = [](double x, double y) { return x / y; };
    auto result = make_jackknife_var_acc(f_ratio, bacc_x, bacc_y);

    ASSERT_GT(result.count(), 1U);
    ASSERT_NEAR(result.naive_mean(), 0.5, 1e-10);
    ASSERT_GE(result.variance(), 0.0);
}

// ================================================================================================
// Edge cases
// ================================================================================================

TEST(SimplemcAccsJackknifeEdge, MinimumTwoSamples) {
    std::vector<double> data = { 1.0, 3.0 };
    auto result = make_jackknife_var_acc(data);

    ASSERT_EQ(result.count(), 2U);
    ASSERT_NEAR(result.naive_mean(), 2.0, 1e-14);

    // Jackknife samples: jk_0 = (2*2 - 1) / 1 = 3, jk_1 = (2*2 - 3) / 1 = 1
    // Mean of jk = 2.0, variance = (2-1)^2 * var_acc.variance()
    ASSERT_NEAR(result.mean(), 2.0, 1e-14);
    // var_acc.variance_of_data() for {3, 1} = (1-0)^2 * 2 / 1 = 2, var = 2/2 = 1
    // jk_var = (2-1)^2 * 1 = 1
    ASSERT_NEAR(result.variance(), 1.0, 1e-14);
}

TEST(SimplemcAccsJackknifeEdge, TooFewSamples) {
    std::vector<double> data = { 1.0 };
    ASSERT_THROW((void)make_jackknife_var_acc(data), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, EmptyRange) {
    std::vector<double> data;
    ASSERT_THROW((void)make_jackknife_var_acc(data), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, MismatchedRanges) {
    std::vector<double> xs = { 1.0, 2.0, 3.0 };
    std::vector<double> ys = { 1.0, 2.0 };
    auto f_ratio = [](double x, double y) { return x / y; };
    ASSERT_THROW((void)make_jackknife_var_acc(f_ratio, xs, ys), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, EmptyBatchAcc) {
    batch_acc<double> bacc(1, 4);
    ASSERT_THROW((void)make_jackknife_var_acc(bacc), simplemc_exception);
}

// ================================================================================================
// Bias-corrected mean consistency
// ================================================================================================

TYPED_TEST(SimplemcAccsJackknife, BiasCorrectedMeanConsistency) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    auto result = make_jackknife_var_acc<A>(data);

    // bias_corrected_mean = original_mean - bias
    T expected_bcm = T(result.original_mean() - result.bias());
    check_near(result.bias_corrected_mean(), expected_bcm, 1e-8);

    // bias_corrected_mean = N * original_mean - (N-1) * mean
    auto n = static_cast<double>(result.count());
    auto nm1 = n - 1.0;
    if constexpr (is_scalar_sample_v<T>) {
        auto expected_bcm2 = n * result.original_mean() - nm1 * result.mean();
        check_near(result.bias_corrected_mean(), expected_bcm2, 1e-8);
    }
}
