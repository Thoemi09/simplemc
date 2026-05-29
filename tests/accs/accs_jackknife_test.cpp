#include "./accs_test_traits.hpp"

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace {

using namespace simplemc;
using namespace test_detail;

} // namespace

// Typed test suite over all 12 accumulator type combinations.
template <typename Tag>
class SimplemcAccsJackknife : public ::testing::Test {};
TYPED_TEST_SUITE(SimplemcAccsJackknife, AllAccTypes);

// Test that jackknife_result satisfies the expected API and concepts.
TYPED_TEST(SimplemcAccsJackknife, ResultSatisfiesConcepts) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    // jackknife_result wraps a covar_acc and exposes the full statistic API.
    using jk_t = jackknife_result<covar_acc<T, A>>;
    static_assert(requires(const jk_t& r) { r.mean(); });
    static_assert(requires(const jk_t& r) { r.variance(); });
    static_assert(requires(const jk_t& r) { r.stderror(); });
    static_assert(requires(const jk_t& r) { r.covariance(); });
    static_assert(requires(const jk_t& r) { r.bias(); });
    static_assert(requires(const jk_t& r) { r.bias_corrected_mean(); });
}

// Test jackknife resampling with f(x) = x.
TYPED_TEST(SimplemcAccsJackknife, Identity) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    auto result = jackknife<A>(std::identity {}, data);

    // mean should equal the data mean
    ASSERT_EQ(result.count(), N);
    check_near(result.naive_mean(), expected_mean<T>(), 1e-10);
    check_near(result.mean(), expected_mean<T>(), 1e-10);

    // bias should be zero, so bias_corrected_mean should equal the data mean
    check_near(result.bias(), make_zero_sample<T>(), 1e-8);
    check_near(result.bias_corrected_mean(), expected_mean<T>(), 1e-8);

    // variance should equal the variance of the mean
    real_vec_t<T> var_data;
    if constexpr (is_complex_sample_v<T>) {
        var_data = expected_var_real_data<T>() + expected_var_imag_data<T>();
    } else {
        var_data = expected_var_data<T>();
    }
    real_vec_t<T> expected_var_mean = var_data / static_cast<double>(N);
    check_near(result.variance(), expected_var_mean, 1e-6);

    // stderror()^2 must equal variance()
    auto se = result.stderror();
    if constexpr (is_scalar_sample_v<T>) {
        ASSERT_GE(se, 0.0);
        check_near(se * se, result.variance(), 1e-8);
    } else {
        real_vec_t<T> se_sq = se.array().square().matrix();
        check_near(se_sq, result.variance(), 1e-8);
    }

    // covariance should equal the covariance of the mean
    if constexpr (is_complex_sample_v<T>) {
        real_mat_t<T> expected_real = (expected_cov_rr<T>() + expected_cov_ii<T>()) / static_cast<double>(N);
        if constexpr (is_scalar_sample_v<T>) {
            check_near(result.covariance(), expected_real, 1e-6);
        } else {
            auto covar = result.covariance();
            real_mat_t<T> covar_re = covar.real();
            real_mat_t<T> covar_im = covar.imag();
            real_mat_t<T> zero = real_mat_t<T>::Zero(result.size(), result.size());
            check_near(covar_re, expected_real, 1e-6);
            check_near(covar_im, zero, 1e-6);
        }
    } else {
        real_mat_t<T> expected_covar_mean = expected_covar_data<T>() / static_cast<double>(N);
        check_near(result.covariance(), expected_covar_mean, 1e-6);
    }
}

// Test jackknife resampling with f(x) = x^2.
TEST(SimplemcAccsJackknifeSingle, SquareTransformation) {
    auto data = make_data<double>();
    const double m = expected_mean<double>();

    auto f_square = [](double x) { return x * x; };
    auto result = jackknife(f_square, data);

    // reference accumulator with jk_i = f((N*m - x_i)/(N-1))
    covar_acc<double> ref;
    for (auto x : data) {
        ref << f_square((N * m - x) / (N - 1.0));
    }
    const double ref_naive = f_square(m);
    const double ref_mean = ref.mean();
    const double ref_var = (N - 1.0) * (N - 1.0) * ref.variance();
    const double ref_bias = (N - 1.0) * (ref_mean - ref_naive);
    const double ref_bcm = N * ref_naive - (N - 1.0) * ref_mean;

    ASSERT_EQ(result.count(), N);
    check_near(result.naive_mean(), ref_naive, 1e-10);
    check_near(result.mean(), ref_mean, 1e-10);
    check_near(result.variance(), ref_var, 1e-8);
    check_near(result.covariance(), ref_var, 1e-8); // scalar: covariance == variance
    check_near(result.bias(), ref_bias, 1e-10);
    check_near(result.bias_corrected_mean(), ref_bcm, 1e-8);
}

// Test jackknife resampling with complex f(z) = z^2.
TEST(SimplemcAccsJackknifeSingle, ComplexSquareTransformation) {
    using Z = std::complex<double>;
    auto data = make_data<Z>();
    const Z m = expected_mean<Z>();

    auto f_square = [](Z z) { return z * z; };
    auto result = jackknife(f_square, data);

    // reference accumulator with jk_i = f((N*m - z_i)/(N-1))
    covar_acc<Z> ref;
    for (auto z : data) {
        ref << f_square((static_cast<double>(N) * m - z) / (N - 1.0));
    }
    const auto ref_naive = f_square(m);
    const auto ref_mean = ref.mean();
    const double ref_var = (N - 1.0) * (N - 1.0) * ref.variance();
    const auto ref_bias = (N - 1.0) * (ref_mean - ref_naive);
    const auto ref_bcm = static_cast<double>(N) * ref_naive - (N - 1.0) * ref_mean;

    ASSERT_EQ(result.count(), N);
    check_near(result.naive_mean(), ref_naive, 1e-10);
    check_near(result.mean(), ref_mean, 1e-10);
    check_near(result.variance(), ref_var, 1e-8);
    check_near(result.covariance(), ref_var, 1e-8); // scalar: covariance == variance
    check_near(result.bias(), ref_bias, 1e-10);
    check_near(result.bias_corrected_mean(), ref_bcm, 1e-8);
}

// Test jackknife resampling with f(x, y) = x/y.
TEST(SimplemcAccsJackknifeMulti, Ratio) {
    // f(x, y) = x/y on xs[i] = i+1, ys[i] = 2*(i+1); since y = 2x exactly, for every i
    // jk_x_i / jk_y_i = 0.5 bit-exactly in IEEE-754, so mean = 0.5 and variance, covariance,
    // and bias are all exactly zero
    auto xs = make_data<double>();
    std::vector<double> ys;
    ys.reserve(N);
    for (auto x : xs) {
        ys.push_back(2.0 * x);
    }

    auto f_ratio = [](double x, double y) { return x / y; };
    auto result = jackknife(f_ratio, xs, ys);

    ASSERT_EQ(result.count(), N);
    check_near(result.naive_mean(), 0.5, 1e-14);
    check_near(result.mean(), 0.5, 1e-14);
    check_near(result.variance(), 0.0, 1e-14);
    check_near(result.covariance(), 0.0, 1e-14);
    check_near(result.bias(), 0.0, 1e-14);
    check_near(result.bias_corrected_mean(), 0.5, 1e-14);
}

// Test jackknife resampling with f(x, y) = {x/y, x*y}.
TEST(SimplemcAccsJackknifeMulti, VectorOutputTransformation) {
    // xs[i] = i+1, ys[i] = 2*(i+1), f(x, y) = {x/y, x*y}; since y = 2x, component 0 is the
    // constant 0.5 (no variance) and component 1 reduces to 2*x_jk^2
    auto xs = make_data<double>();
    std::vector<double> ys;
    ys.reserve(N);
    for (auto x : xs) {
        ys.push_back(2.0 * x);
    }
    const double m_x = expected_mean<double>();

    auto f_vec = [](double x, double y) {
        Eigen::Vector2d v;
        v << x / y, x * y;
        return v;
    };
    auto result = jackknife(f_vec, xs, ys);

    // reference: jk_i has component 0 = 0.5 exactly and component 1 = 2 * jk_x_i^2 where
    // jk_x_i = (N*m_x - x_i)/(N-1)
    covar_acc<Eigen::Vector2d> ref;
    for (auto x : xs) {
        const double x_jk = (N * m_x - x) / (N - 1.0);
        Eigen::Vector2d v;
        v << 0.5, 2.0 * x_jk * x_jk;
        ref << v;
    }
    Eigen::Vector2d ref_naive;
    ref_naive << 0.5, 2.0 * m_x * m_x;
    Eigen::Vector2d ref_mean = ref.mean();
    Eigen::Vector2d ref_var = (N - 1.0) * (N - 1.0) * ref.variance();
    Eigen::Vector2d ref_bias = (N - 1.0) * (ref_mean - ref_naive);
    Eigen::Vector2d ref_bcm = N * ref_naive - (N - 1.0) * ref_mean;

    ASSERT_EQ(result.count(), N);
    check_near(result.naive_mean(), ref_naive, 1e-10);
    check_near(result.mean(), ref_mean, 1e-10);
    check_near(result.variance(), ref_var, 1e-8);
    check_near(result.bias(), ref_bias, 1e-10);
    check_near(result.bias_corrected_mean(), ref_bcm, 1e-8);
}

// Test jackknife resampling on a batch_acc with f(x) = x.
TYPED_TEST(SimplemcAccsJackknife, BatchAccIdentity) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    // accumulate the data into a batch_acc; jackknife operates on the batch means rather
    // than the raw samples, so the result should match a range-based jackknife on those means
    auto data = make_data<T>();
    constexpr std::size_t m_b = 4;
    auto bacc = make_empty_batch<batch_acc<T, A>, T>(m_b);
    for (const auto& x : data) {
        bacc << x;
    }
    auto result = jackknife(std::identity {}, bacc);

    // reference: range-based jackknife on the same batch means
    auto bs = bacc.batches();
    std::vector<T> means;
    means.reserve(bs.size());
    for (const auto& b : bs) {
        means.push_back(b.mean());
    }
    auto result_range = jackknife<A>(std::identity {}, means);

    ASSERT_EQ(result.count(), static_cast<decltype(result.count())>(bacc.num_full_batches()));
    check_near(result.naive_mean(), result_range.naive_mean(), 1e-10);
    check_near(result.mean(), result_range.mean(), 1e-10);
    check_near(result.variance(), result_range.variance(), 1e-10);
    check_near(result.stderror(), result_range.stderror(), 1e-10);
    check_near(result.covariance(), result_range.covariance(), 1e-10);
    check_near(result.bias(), result_range.bias(), 1e-10);
    check_near(result.bias_corrected_mean(), result_range.bias_corrected_mean(), 1e-8);
}

// Test jackknife resampling on batch_acc's with f(x, y) = x/y.
TEST(SimplemcAccsJackknifeBatch, MultiBatchRatio) {
    // batched xs[i] = i+1, ys[i] = 2*(i+1); since y = 2x exactly and each batch mean of y
    // is 2x the batch mean of x, jk_x/jk_y = 0.5 bit-exactly per batch, so mean = 0.5 and
    // variance, covariance, and bias are all exactly zero
    constexpr int n = 200;
    constexpr std::size_t m_b = 4;

    batch_acc<double> bacc_x(1, m_b);
    batch_acc<double> bacc_y(1, m_b);

    for (int i = 0; i < n; ++i) {
        bacc_x << static_cast<double>(i + 1);
        bacc_y << 2.0 * static_cast<double>(i + 1);
    }

    auto f_ratio = [](double x, double y) { return x / y; };
    auto result = jackknife(f_ratio, bacc_x, bacc_y);

    ASSERT_EQ(result.count(), static_cast<decltype(result.count())>(bacc_x.num_full_batches()));
    check_near(result.naive_mean(), 0.5, 1e-14);
    check_near(result.mean(), 0.5, 1e-14);
    check_near(result.variance(), 0.0, 1e-14);
    check_near(result.covariance(), 0.0, 1e-14);
    check_near(result.bias(), 0.0, 1e-14);
    check_near(result.bias_corrected_mean(), 0.5, 1e-14);
}

// Test jackknife resampling with only two samples and f(x) = x.
TEST(SimplemcAccsJackknifeEdge, MinimumTwoSamples) {
    // identity on data = {1, 3}; jackknife samples: jk_0 = (2*2 - 1)/1 = 3, jk_1 = (2*2 - 3)/1 = 1
    // var_acc.variance_of_data() for {3, 1} = 2, var = 2/N = 1, jk_var = (N-1)^2 * 1 = 1
    // bias = (N-1)*(mean - naive) = 0, bias_corrected_mean = N*naive - (N-1)*mean = 2
    std::vector<double> data = { 1.0, 3.0 };
    auto result = jackknife(std::identity {}, data);

    // compare with analytical reference values
    ASSERT_EQ(result.count(), 2U);
    check_near(result.naive_mean(), 2.0, 1e-14);
    check_near(result.mean(), 2.0, 1e-14);
    check_near(result.variance(), 1.0, 1e-14);
    check_near(result.covariance(), 1.0, 1e-14); // scalar: covariance == variance
    check_near(result.stderror(), 1.0, 1e-14);
    check_near(result.bias(), 0.0, 1e-14);
    check_near(result.bias_corrected_mean(), 2.0, 1e-14);
}

// Test that jackknife resampling throws when the input is invalid.
TEST(SimplemcAccsJackknifeEdge, TooFewSamples) {
    std::vector<double> data = { 1.0 };
    ASSERT_THROW((void)jackknife(std::identity {}, data), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, EmptyRange) {
    std::vector<double> data;
    ASSERT_THROW((void)jackknife(std::identity {}, data), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, MismatchedRanges) {
    std::vector<double> xs = { 1.0, 2.0, 3.0 };
    std::vector<double> ys = { 1.0, 2.0 };
    auto f_ratio = [](double x, double y) { return x / y; };
    ASSERT_THROW((void)jackknife(f_ratio, xs, ys), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, EmptyBatchAcc) {
    batch_acc<double> bacc(1, 4);
    ASSERT_THROW((void)jackknife(std::identity {}, bacc), simplemc_exception);
}

TEST(SimplemcAccsJackknifeEdge, MismatchedBatchAccs) {
    // two batch_acc's with the same m_b but very different numbers of samples produce
    // different num_full_batches; the multi-batch jackknife overload must reject this
    constexpr std::size_t m_b = 4;
    batch_acc<double> bacc_x(1, m_b);
    batch_acc<double> bacc_y(1, m_b);
    for (int i = 0; i < 10; ++i) {
        bacc_x << static_cast<double>(i + 1);
    }
    for (int i = 0; i < 200; ++i) {
        bacc_y << static_cast<double>(i + 1);
    }
    ASSERT_NE(bacc_x.num_full_batches(), bacc_y.num_full_batches());

    auto f_sum = [](double x, double y) { return x + y; };
    ASSERT_THROW((void)jackknife(f_sum, bacc_x, bacc_y), simplemc_exception);
}

// Test jackknife resampling with different accumulating algorithms.
TEST(SimplemcAccsJackknifeAlgAgree, WelfordVsStandard) {
    std::vector<double> data;
    data.reserve(N);
    for (int i = 0; i < N; ++i) {
        data.push_back(static_cast<double>(i + 1) / N);
    }

    auto f_square = [](double x) { return x * x; };
    auto r_w = jackknife<varalg::welford>(f_square, data);
    auto r_s = jackknife<varalg::standard>(f_square, data);

    constexpr double tol = 1e-12;
    check_near(r_w.naive_mean(), r_s.naive_mean(), tol);
    check_near(r_w.mean(), r_s.mean(), tol);
    check_near(r_w.variance(), r_s.variance(), tol);
    check_near(r_w.stderror(), r_s.stderror(), tol);
    check_near(r_w.covariance(), r_s.covariance(), tol);
    check_near(r_w.bias(), r_s.bias(), tol);
    check_near(r_w.bias_corrected_mean(), r_s.bias_corrected_mean(), tol);
}
