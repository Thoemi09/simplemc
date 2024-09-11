#include "./accs_fixture.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>

#include <Eigen/Dense>
#include <range/v3/view/enumerate.hpp>

#include <complex>
#include <tuple>
#include <utility>

// Accumulate mean data.
template <simplemc::accs::varalg A, typename S>
[[nodiscard]] auto mean_data(const S& sp, const Eigen::VectorX<typename S::value_type>& shift) {
    using vec_type = Eigen::VectorX<typename S::value_type>;
    vec_type res = vec_type::Zero(S::state_size);
    for (const auto& [i, s] : ranges::views::enumerate(sp.samples)) {
        if constexpr (A == simplemc::accs::varalg::standard) {
            res += s.matrix() - shift;
        } else {
            res = res + (s.matrix() - shift - res) / static_cast<double>(i + 1);
        }
    }
    return res;
}

// Accumulate mean and covariance double data.
template <simplemc::accs::varalg A, typename S>
[[nodiscard]] auto accumulate_data(const S& sp, const Eigen::VectorXd& shift) {
    Eigen::VectorXd mdata = Eigen::VectorXd::Zero(S::state_size);
    Eigen::MatrixXd cdata = Eigen::MatrixXd::Zero(S::state_size, S::state_size);
    for (const auto& [i, s] : ranges::views::enumerate(sp.samples)) {
        if constexpr (A == simplemc::accs::varalg::standard) {
            const auto tmp = s.matrix() - shift;
            mdata += tmp;
            cdata += tmp * tmp.transpose();
        } else {
            const auto tmp = (s.matrix() - shift - mdata).eval();
            mdata += tmp / static_cast<double>(i + 1);
            cdata += tmp * (s.matrix() - shift - mdata).transpose();
        }
    }
    return std::make_pair(mdata, cdata);
}

// Print analytic results of the stochastic processes in the fixture.
TEST_F(SimplemcAccs, PrintAnalyticResults) {
    print_analytic_results();
}

// Test the make_nans functions.
TEST_F(SimplemcAccs, UtilsMakeNans) {
    using namespace simplemc::accs;
    auto vec_d = nans_vector<Eigen::VectorXd>(10);
    auto vec_sd = nans_vector<Eigen::Vector<double, 10>>();
    auto vec_c = nans_vector<Eigen::VectorXcd>(10);
    auto vec_sc = nans_vector<Eigen::Vector<std::complex<double>, 10>>();
    for (int i = 0; i < 10; ++i) {
        check_isnan(vec_d[i]);
        check_isnan(vec_sd[i]);
        check_isnan(vec_c[i]);
        check_isnan(vec_sc[i]);
    }

    auto mat_d = nans_matrix<Eigen::MatrixXd>(5, 5);
    auto mat_sd = nans_matrix<Eigen::Matrix<double, 5, 5>>();
    auto mat_c = nans_matrix<Eigen::MatrixXcd>(5, 5);
    auto mat_sc = nans_matrix<Eigen::Matrix<std::complex<double>, 5, 5>>();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            check_isnan(mat_d(i, j));
            check_isnan(mat_sd(i, j));
            check_isnan(mat_c(i, j));
            check_isnan(mat_sc(i, j));
        }
    }
}

// Test the mean function.
TEST_F(SimplemcAccs, UtilsMean) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using cplx_vec_type = Eigen::VectorXcd;
    const double tol = 1e-10;
    const dbl_vec_type sm_d = sample_mean(sp_d);
    const dbl_vec_type am_d = analytic_mean(sp_d).matrix();
    const dbl_vec_type zero_d = dbl_vec_type::Zero(size);
    const cplx_vec_type sm_c = sample_mean(sp_c);
    const cplx_vec_type am_c = analytic_mean(sp_c).matrix();
    const cplx_vec_type zero_c = cplx_vec_type::Zero(size);

    // standard, double, zero shift
    auto mdata_d = mean_data<varalg::standard>(sp_d, zero_d);
    check_range_near(mean<varalg::standard>(mdata_d, sp_d.total_count, zero_d), sm_d, tol);

    // standard, double, finite shift
    mdata_d = mean_data<varalg::standard>(sp_d, am_d);
    check_range_near(mean<varalg::standard>(mdata_d, sp_d.total_count, am_d), sm_d, tol);

    // welford, double, zero shift
    mdata_d = mean_data<varalg::welford>(sp_d, zero_d);
    check_range_near(mean<varalg::welford>(mdata_d, sp_d.total_count, zero_d), sm_d, tol);

    // welford, double, finite shift
    mdata_d = mean_data<varalg::welford>(sp_d, am_d);
    check_range_near(mean<varalg::welford>(mdata_d, sp_d.total_count, am_d), sm_d, tol);

    // standard, complex, zero shift
    auto mdata_c = mean_data<varalg::standard>(sp_c, zero_c);
    check_range_near(mean<varalg::standard>(mdata_c, sp_c.total_count, zero_c), sm_c, tol);

    // standard, complex, finite shift
    mdata_c = mean_data<varalg::standard>(sp_c, am_c);
    check_range_near(mean<varalg::standard>(mdata_c, sp_c.total_count, am_c), sm_c, tol);

    // welford, complex, zero shift
    mdata_c = mean_data<varalg::welford>(sp_c, zero_c);
    check_range_near(mean<varalg::welford>(mdata_c, sp_c.total_count, zero_c), sm_c, tol);

    // welford, complex, finite shift
    mdata_c = mean_data<varalg::welford>(sp_c, am_c);
    check_range_near(mean<varalg::welford>(mdata_c, sp_c.total_count, am_c), sm_c, tol);
}

// Test the diag_covariance function.
TEST_F(SimplemcAccs, UtilsDiagCovariance) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    const double tol = 1e-10;
    const dbl_vec_type sv_d = sample_variance(sp_d);
    const dbl_vec_type am_d = analytic_mean(sp_d).matrix();
    const dbl_vec_type zero_d = dbl_vec_type::Zero(size);

    // standard, double, zero shift
    auto [mdata_d, cdata_d] = accumulate_data<varalg::standard>(sp_d, zero_d);
    check_range_near(
        diag_covariance<varalg::standard>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d,
        tol);

    // standard, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<varalg::standard>(sp_d, am_d);
    check_range_near(
        diag_covariance<varalg::standard>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d,
        tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<varalg::welford>(sp_d, zero_d);
    check_range_near(
        diag_covariance<varalg::welford>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d,
        tol);

    // welford, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<varalg::welford>(sp_d, am_d);
    check_range_near(
        diag_covariance<varalg::welford>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d,
        tol);
}

// Test the covariance function.
TEST_F(SimplemcAccs, UtilsCovariance) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using dbl_mat_type = Eigen::MatrixXd;
    const double tol = 1e-10;
    const dbl_mat_type scv_d = sample_covariance(sp_d);
    const dbl_vec_type am_d = analytic_mean(sp_d).matrix();
    const dbl_vec_type zero_d = dbl_vec_type::Zero(size);

    // standard, double, zero shift
    using simplemc::make_span;
    auto [mdata_d, cdata_d] = accumulate_data<varalg::standard>(sp_d, zero_d);
    check_range_near(
        make_span(covariance<varalg::standard>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // standard, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<varalg::standard>(sp_d, am_d);
    check_range_near(
        make_span(covariance<varalg::standard>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<varalg::welford>(sp_d, zero_d);
    check_range_near(
        make_span(covariance<varalg::welford>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // welford, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<varalg::welford>(sp_d, am_d);
    check_range_near(
        make_span(covariance<varalg::welford>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);
}
