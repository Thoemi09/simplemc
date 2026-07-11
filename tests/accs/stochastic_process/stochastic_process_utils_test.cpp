// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "./stochastic_process_fixture.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/ranges.hpp>

#include <Eigen/Dense>

#include <array>
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
TEST_F(SimplemcAccsStochasticProcess, PrintAnalyticResults) {
    print_analytic_results();
}

// Test the mean function.
TEST_F(SimplemcAccsStochasticProcess, UtilsMean) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    using cplx_vec_type = Eigen::VectorXcd;
    const dbl_vec_type sm_d = sample_mean(sp_d);
    const cplx_vec_type sm_c = sample_mean(sp_c);

    // standard, double, zero shift
    auto mdata_d = mean_data<standard>(sp_d);
    check_near(mean<standard>(mdata_d, sp_d.total_count), sm_d, tol);

    // welford, double, zero shift
    mdata_d = mean_data<welford>(sp_d);
    check_near(mean<welford>(mdata_d, sp_d.total_count), sm_d, tol);

    // standard, complex, zero shift
    auto mdata_c = mean_data<standard>(sp_c);
    check_near(mean<standard>(mdata_c, sp_c.total_count), sm_c, tol);

    // welford, complex, zero shift
    mdata_c = mean_data<welford>(sp_c);
    check_near(mean<welford>(mdata_c, sp_c.total_count), sm_c, tol);
}

// Test the diag_covariance function.
TEST_F(SimplemcAccsStochasticProcess, UtilsDiagCovariance) {
    using namespace simplemc::accs;
    using dbl_vec_type = Eigen::VectorXd;
    const dbl_vec_type sv_d = sample_variance(sp_d);

    // standard, double, zero shift
    auto [mdata_d, cdata_d] = accumulate_data<standard>(sp_d);
    check_near(
        diag_covariance<standard>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_near(
        diag_covariance<welford>(mdata_d, mdata_d, dbl_vec_type(cdata_d.diagonal()), sp_d.total_count), sv_d, tol);
}

// Test the covariance function.
TEST_F(SimplemcAccsStochasticProcess, UtilsCovariance) {
    using namespace simplemc::accs;
    using dbl_mat_type = Eigen::MatrixXd;
    const dbl_mat_type scv_d = sample_covariance(sp_d);

    // standard, double, zero shift
    using simplemc::make_span;
    auto [mdata_d, cdata_d] = accumulate_data<standard>(sp_d);
    check_near(make_span(covariance<standard>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);

    // welford, double, zero shift
    std::tie(mdata_d, cdata_d) = accumulate_data<welford>(sp_d);
    check_near(make_span(covariance<welford>(mdata_d, mdata_d, cdata_d, sp_d.total_count)), make_span(scv_d), tol);
}

// Test tau with real stochastic process data using blocking method.
TEST_F(SimplemcAccsStochasticProcess, UtilsTauBlocking) {
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
        check_near(simplemc::make_span(computed_tau_v), simplemc::make_span(Eigen::MatrixXd(taus_v[1])), tol);

        // similarly for covariance
        const auto computed_tau_c =
            tau(Eigen::MatrixXd(s_naive_c.matrix()), Eigen::MatrixXd(s_blocked_c_1.matrix()), b1);
        check_near(simplemc::make_span(computed_tau_c), simplemc::make_span(Eigen::MatrixXd(taus_c[1].matrix())), tol);
    }
}
