#include "./accs_fixture.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>

#include <Eigen/Dense>

#include <tuple>
#include <utility>

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
TEST_F(SimplemcAccs, PrintAnalyticResults) {
    print_analytic_results();
}

// Test the mean function.
TEST_F(SimplemcAccs, UtilsMean) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using cplx_vec_type = Eigen::VectorXcd;
    const dbl_vec_type sm_d = sample_mean(sp_d);
    const cplx_vec_type sm_c = sample_mean(sp_c);

    // standard, double, zero shift
    auto mdata_d = mean_data<standard>(sp_d);
    check_range_near(mean<standard>(mdata_d, sp_d.total_count), sm_d, tol);

    // standard, double, finite shift
    mdata_d = mean_data<standard>(sp_d);
    check_range_near(mean<standard>(mdata_d, sp_d.total_count), sm_d, tol);

    // welford, double, zero shift
    mdata_d = mean_data<welford>(sp_d);
    check_range_near(mean<welford>(mdata_d, sp_d.total_count), sm_d, tol);

    // welford, double, finite shift
    mdata_d = mean_data<welford>(sp_d);
    check_range_near(mean<welford>(mdata_d, sp_d.total_count), sm_d, tol);

    // standard, complex, zero shift
    auto mdata_c = mean_data<standard>(sp_c);
    check_range_near(mean<standard>(mdata_c, sp_c.total_count), sm_c, tol);

    // standard, complex, finite shift
    mdata_c = mean_data<standard>(sp_c);
    check_range_near(mean<standard>(mdata_c, sp_c.total_count), sm_c, tol);

    // welford, complex, zero shift
    mdata_c = mean_data<welford>(sp_c);
    check_range_near(mean<welford>(mdata_c, sp_c.total_count), sm_c, tol);

    // welford, complex, finite shift
    mdata_c = mean_data<welford>(sp_c);
    check_range_near(mean<welford>(mdata_c, sp_c.total_count), sm_c, tol);
}

// Test the diag_covariance function.
TEST_F(SimplemcAccs, UtilsDiagCovariance) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    const dbl_vec_type sv_d = sample_variance(sp_d);

    // standard, double, zero shift
    auto [mdata_d, cdata_d] = accumulate_data<standard>(sp_d);
    check_range_near(
        diag_covariance<standard>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);

    // standard, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<standard>(sp_d);
    check_range_near(
        diag_covariance<standard>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_range_near(
        diag_covariance<welford>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);

    // welford, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_range_near(
        diag_covariance<welford>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);
}

// Test the covariance function.
TEST_F(SimplemcAccs, UtilsCovariance) {
    using namespace simplemc::accs;
    using dbl_mat_type = Eigen::MatrixXd;
    const dbl_mat_type scv_d = sample_covariance(sp_d);

    // standard, double, zero shift
    using simplemc::make_span;
    auto [mdata_d, cdata_d] = accumulate_data<standard>(sp_d);
    check_range_near(
        make_span(covariance<standard>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // standard, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<standard>(sp_d);
    check_range_near(
        make_span(covariance<standard>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_range_near(
        make_span(covariance<welford>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // welford, double, finite shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_range_near(
        make_span(covariance<welford>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);
}
