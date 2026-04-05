#include "./accs_fixture_advanced.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/ranges.hpp>

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;
constexpr double tol = 1e-10;

} // namespace

// Accumulate mean data.
template <simplemc::varalg A, typename S>
[[nodiscard]] auto mean_data(const S& sp) {
    using vec_type = Eigen::VectorX<typename S::value_type>;
    vec_type res = vec_type::Zero(S::state_size);
    for (int i = 0; const auto& s : sp.samples) {
        if constexpr (A == standard) {
            res += s.matrix();
        } else {
            res = res + (s.matrix() - res) / static_cast<double>(i++ + 1);
        }
    }
    return res;
}

// Accumulate mean and covariance double data.
template <simplemc::varalg A, typename S>
[[nodiscard]] auto accumulate_data(const S& sp) {
    Eigen::VectorXd mdata = Eigen::VectorXd::Zero(S::state_size);
    Eigen::MatrixXd cdata = Eigen::MatrixXd::Zero(S::state_size, S::state_size);
    for (int i = 0; const auto& s : sp.samples) {
        if constexpr (A == standard) {
            const auto tmp = s.matrix();
            mdata += tmp;
            cdata += tmp * tmp.transpose();
        } else {
            const auto tmp = (s.matrix() - mdata).eval();
            mdata += tmp / static_cast<double>(i++ + 1);
            cdata += tmp * (s.matrix() - mdata).transpose();
        }
    }
    return std::make_pair(mdata, cdata);
}

// Print analytic results of the stochastic processes in the fixture.
TEST_F(SimplemcAccsAdvanced, PrintAnalyticResults) {
    print_analytic_results();
}

// Test the mean function.
TEST_F(SimplemcAccsAdvanced, UtilsMean) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using cplx_vec_type = Eigen::VectorXcd;
    const dbl_vec_type sm_d = sample_mean(sp_d);
    const cplx_vec_type sm_c = sample_mean(sp_c);

    // standard, double, zero shift
    auto mdata_d = mean_data<standard>(sp_d);
    check_range_near(mean<standard>(mdata_d, sp_d.total_count), sm_d, tol);

    // welford, double, zero shift
    mdata_d = mean_data<welford>(sp_d);
    check_range_near(mean<welford>(mdata_d, sp_d.total_count), sm_d, tol);

    // standard, complex, zero shift
    auto mdata_c = mean_data<standard>(sp_c);
    check_range_near(mean<standard>(mdata_c, sp_c.total_count), sm_c, tol);

    // welford, complex, zero shift
    mdata_c = mean_data<welford>(sp_c);
    check_range_near(mean<welford>(mdata_c, sp_c.total_count), sm_c, tol);
}

// Test the diag_covariance function.
TEST_F(SimplemcAccsAdvanced, UtilsDiagCovariance) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    const dbl_vec_type sv_d = sample_variance(sp_d);

    // standard, double, zero shift
    auto [mdata_d, cdata_d] = accumulate_data<standard>(sp_d);
    check_range_near(
        diag_covariance<standard>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_range_near(
        diag_covariance<welford>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);
}

// Test the covariance function.
TEST_F(SimplemcAccsAdvanced, UtilsCovariance) {
    using namespace simplemc::accs;
    using dbl_mat_type = Eigen::MatrixXd;
    const dbl_mat_type scv_d = sample_covariance(sp_d);

    // standard, double, zero shift
    using simplemc::make_span;
    auto [mdata_d, cdata_d] = accumulate_data<standard>(sp_d);
    check_range_near(
        make_span(covariance<standard>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_range_near(
        make_span(covariance<welford>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);
}

// Test random sample concepts and helper functions.
TEST_F(SimplemcAccsAdvanced, UtilsRandomSamples) {
    // sample_type concept
    static_assert(simplemc::sample_type<double>);
    static_assert(simplemc::sample_type<std::complex<double>>);
    static_assert(simplemc::sample_type<Eigen::VectorXd>);
    static_assert(!simplemc::sample_type<int>);
    static_assert(!simplemc::sample_type<Eigen::MatrixXd>);

    // sample_range concept
    static_assert(simplemc::sample_range<std::vector<double>>);
    static_assert(simplemc::sample_range<std::array<std::complex<double>, 5>>);
    static_assert(simplemc::sample_range<std::vector<Eigen::VectorXd>>);
    auto view = simplemc::ranges::transform_view(sp_d.samples, [](const auto& s) { return Eigen::VectorXd { s }; });
    static_assert(simplemc::sample_range<decltype(view)>);

    // random sample size
    EXPECT_EQ(simplemc::detail::random_sample_size(1.0), 1);
    EXPECT_EQ(simplemc::detail::random_sample_size(Eigen::Vector3d()), 3);
    EXPECT_EQ(simplemc::detail::random_sample_size(Eigen::VectorXcd(5)), 5);

    // zero sample
    EXPECT_EQ(simplemc::detail::zero_sample(1.0), 0.0);
    EXPECT_EQ(simplemc::detail::zero_sample(std::complex<double>(1.0, -1.0)), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(simplemc::detail::zero_sample(Eigen::Vector3d()), Eigen::Vector3d::Zero());
    EXPECT_EQ(simplemc::detail::zero_sample(Eigen::VectorXcd(5)), Eigen::VectorXcd::Zero(5));
}

// Test edge cases for mean, diag_covariance, and covariance functions.
TEST_F(SimplemcAccsAdvanced, UtilsEdgeCases) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using dbl_mat_type = Eigen::MatrixXd;
    auto is_nan = [](double x) { return std::isnan(x); };

    const dbl_vec_type mdata = dbl_vec_type::Ones(3);
    const dbl_vec_type cdata_v = dbl_vec_type::Ones(3);
    const dbl_mat_type cdata_m = dbl_mat_type::Ones(3, 3);

    // mean with n == 0 should return NaNs
    auto m0 = mean<standard>(mdata, 0);
    EXPECT_EQ(m0.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(m0, is_nan));
    m0 = mean<welford>(mdata, 0);
    EXPECT_TRUE(std::ranges::all_of(m0, is_nan));

    // diag_covariance with n == 0 and n == 1 should return NaNs
    auto dc0 = diag_covariance<standard>(mdata, mdata, cdata_v, 0);
    EXPECT_EQ(dc0.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(dc0, is_nan));
    auto dc1 = diag_covariance<standard>(mdata, mdata, cdata_v, 1);
    EXPECT_TRUE(std::ranges::all_of(dc1, is_nan));
    dc0 = diag_covariance<welford>(mdata, mdata, cdata_v, 0);
    EXPECT_TRUE(std::ranges::all_of(dc0, is_nan));
    dc1 = diag_covariance<welford>(mdata, mdata, cdata_v, 1);
    EXPECT_TRUE(std::ranges::all_of(dc1, is_nan));

    // covariance with n == 0 and n == 1 should return NaNs
    auto cv0 = covariance<standard>(mdata, mdata, cdata_m, 0);
    EXPECT_EQ(cv0.rows(), 3);
    EXPECT_EQ(cv0.cols(), 3);
    EXPECT_TRUE(std::ranges::all_of(simplemc::make_span(cv0), is_nan));
    auto cv1 = covariance<standard>(mdata, mdata, cdata_m, 1);
    EXPECT_TRUE(std::ranges::all_of(simplemc::make_span(cv1), is_nan));
    cv0 = covariance<welford>(mdata, mdata, cdata_m, 0);
    EXPECT_TRUE(std::ranges::all_of(simplemc::make_span(cv0), is_nan));
    cv1 = covariance<welford>(mdata, mdata, cdata_m, 1);
    EXPECT_TRUE(std::ranges::all_of(simplemc::make_span(cv1), is_nan));
}

// Test the tau function for scalar values.
TEST_F(SimplemcAccsAdvanced, UtilsTauScalar) {
    using namespace simplemc::accs;

    // tau = 0.5 * (s_blocked * b / s_naive - 1)
    // for block size 1 with equal variances: tau = 0.5 * (1 - 1) = 0
    EXPECT_DOUBLE_EQ(tau(1.0, 1.0, 1), 0.0);

    // for s_blocked = 2.0, s_naive = 1.0, b = 2: tau = 0.5 * (2*2/1 - 1) = 1.5
    EXPECT_DOUBLE_EQ(tau(1.0, 2.0, 2), 1.5);

    // for s_blocked = 1.0, s_naive = 2.0, b = 4: tau = 0.5 * (1*4/2 - 1) = 0.5
    EXPECT_DOUBLE_EQ(tau(2.0, 1.0, 4), 0.5);

    // general formula verification
    const double s_naive = 3.5;
    const double s_blocked = 7.0;
    const std::uint64_t b = 10;
    const double expected = 0.5 * (s_blocked * static_cast<double>(b) / s_naive - 1.0);
    EXPECT_DOUBLE_EQ(tau(s_naive, s_blocked, b), expected);
}

// Test the tau function for matrix/vector values.
TEST_F(SimplemcAccsAdvanced, UtilsTauMatrix) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using dbl_mat_type = Eigen::MatrixXd;

    // test with variance vectors (treated as 1-column matrices by the template)
    dbl_vec_type s_naive_v(3);
    s_naive_v << 1.0, 2.0, 4.0;
    dbl_vec_type s_blocked_v(3);
    s_blocked_v << 2.0, 4.0, 8.0;
    const std::uint64_t b_v = 2;

    // expected: tau = 0.5 * (s_blocked * b / s_naive - 1) for each element
    // [0]: 0.5 * (2*2/1 - 1) = 1.5
    // [1]: 0.5 * (4*2/2 - 1) = 1.5
    // [2]: 0.5 * (8*2/4 - 1) = 1.5
    dbl_vec_type expected_v(3);
    expected_v << 1.5, 1.5, 1.5;
    check_range_near(tau(s_naive_v, s_blocked_v, b_v), expected_v, tol);

    // test with covariance matrices
    dbl_mat_type s_naive_m(2, 2);
    s_naive_m << 1.0, 0.5, 0.5, 2.0;
    dbl_mat_type s_blocked_m(2, 2);
    s_blocked_m << 4.0, 1.0, 1.0, 4.0;
    const std::uint64_t b_m = 4;

    // expected: tau = 0.5 * (s_blocked * b / s_naive - 1) for each element
    // [0,0]: 0.5 * (4*4/1 - 1) = 7.5
    // [0,1]: 0.5 * (1*4/0.5 - 1) = 3.5
    // [1,0]: 0.5 * (1*4/0.5 - 1) = 3.5
    // [1,1]: 0.5 * (4*4/2 - 1) = 3.5
    dbl_mat_type expected_m(2, 2);
    expected_m << 7.5, 3.5, 3.5, 3.5;
    check_range_near(simplemc::make_span(tau(s_naive_m, s_blocked_m, b_m)), simplemc::make_span(expected_m), tol);
}

// Test tau with real stochastic process data using blocking method.
TEST_F(SimplemcAccsAdvanced, UtilsTauBlocking) {
    using namespace simplemc::accs;

    // use the blocking_autocorr function from the fixture to get blocked samples
    const auto [bsps, taus_v, taus_c] = blocking_autocorr(sp_d, 2);

    // verify that tau function reproduces the blocking results
    if (bsps.size() >= 2) {
        const auto s_naive_v = sample_variance(bsps[0]);
        const auto s_naive_c = sample_covariance(bsps[0]);

        // Check at block size 2 (index 1)
        const auto s_blocked_v_1 = sample_variance(bsps[1]);
        const auto s_blocked_c_1 = sample_covariance(bsps[1]);
        const std::uint64_t b1 = 2;

        // tau_v[1] = 0.5 * (s_blocked * b / s_naive - 1)
        const auto computed_tau_v = tau(Eigen::MatrixXd(s_naive_v), Eigen::MatrixXd(s_blocked_v_1), b1);
        check_range_near(simplemc::make_span(computed_tau_v), simplemc::make_span(Eigen::MatrixXd(taus_v[1])), tol);

        // similarly for covariance
        const auto computed_tau_c =
            tau(Eigen::MatrixXd(s_naive_c.matrix()), Eigen::MatrixXd(s_blocked_c_1.matrix()), b1);
        check_range_near(
            simplemc::make_span(computed_tau_c), simplemc::make_span(Eigen::MatrixXd(taus_c[1].matrix())), tol);
    }
}
