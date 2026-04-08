#ifndef SIMPLEMC_TESTS_ACCS_FIXTURE_HPP
#define SIMPLEMC_TESTS_ACCS_FIXTURE_HPP

#include "../test_utils.hpp"

#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <complex>
#include <vector>

// Test fixture for the accumulator tests using small, deterministic data sets with hand-computed
// expected values.
class SimplemcAccs : public ::testing::Test {
protected:
    using cplx = std::complex<double>;

    void SetUp() override {
        using namespace std::literals::complex_literals;

        // data set: 5 scalar double samples.
        dbl_5 = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        dbl_5_n = 5;
        dbl_5_mean = 3.0;
        dbl_5_var_data = 2.5; // Bessel-corrected
        dbl_5_var_mean = 0.5; // var_data / n

        // data set: 5 scalar complex samples.
        cplx_5 = { 1.0 + 2.0i, 3.0 + 1.0i, 2.0 + 4.0i, 4.0 + 3.0i, 5.0 + 0.0i };
        cplx_5_n = 5;
        cplx_5_mean = 3.0 + 2.0i;
        cplx_5_var_re = 2.5;
        cplx_5_var_im = 2.5;
        cplx_5_cov_re_im = -1.25;

        // data set: 4 double vectors of size 3.
        vec_d_data.resize(4);
        vec_d_data[0] = Eigen::Vector3d(1.0, 4.0, 2.0);
        vec_d_data[1] = Eigen::Vector3d(3.0, 6.0, 4.0);
        vec_d_data[2] = Eigen::Vector3d(2.0, 5.0, 3.0);
        vec_d_data[3] = Eigen::Vector3d(4.0, 5.0, 1.0);
        vec_d_n = 4;
        vec_d_mean = Eigen::Vector3d(2.5, 5.0, 2.5);
        vec_d_var = Eigen::Vector3d(5.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0);

        vec_d_cov = Eigen::Matrix3d::Zero();
        vec_d_cov.row(0) = Eigen::Vector3d(5.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0);
        vec_d_cov.row(1) = Eigen::Vector3d(2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
        vec_d_cov.row(2) = Eigen::Vector3d(-1.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0);

        // data set: 3 complex vectors of size 2.
        vec_c_data.resize(3);
        vec_c_data[0] = Eigen::Vector2cd(1.0 + 1.0i, 2.0 - 1.0i);
        vec_c_data[1] = Eigen::Vector2cd(2.0 + 0.0i, 3.0 + 2.0i);
        vec_c_data[2] = Eigen::Vector2cd(3.0 - 1.0i, 1.0 + 0.0i);
        vec_c_n = 3;
        vec_c_mean = Eigen::Vector2cd(2.0 + 0.0i, 2.0 + 1.0 / 3.0 * 1.0i);

        // variance (diagonal of covariance).
        vec_c_var_re = Eigen::Vector2d(1.0, 1.0);
        vec_c_var_im = Eigen::Vector2d(1.0, 7.0 / 3.0);
        vec_c_cov_re_im_diag = Eigen::Vector2d(-1.0, 1.0);

        // covariance matrices (2x2).
        vec_c_cov_rr = Eigen::Matrix2d::Zero();
        vec_c_cov_rr << 1.0, -0.5, -0.5, 1.0;

        vec_c_cov_ii = Eigen::Matrix2d::Zero();
        vec_c_cov_ii << 1.0, -0.5, -0.5, 7.0 / 3.0;

        vec_c_cov_ri = Eigen::Matrix2d::Zero();
        vec_c_cov_ri << -1.0, 0.5, 0.5, 1.0;
    }

    // scalar double data (5 samples)
    std::vector<double> dbl_5;
    int dbl_5_n {};
    double dbl_5_mean {};
    double dbl_5_var_data {};
    double dbl_5_var_mean {};

    // scalar complex data (5 samples)
    std::vector<cplx> cplx_5;
    int cplx_5_n {};
    cplx cplx_5_mean {};
    double cplx_5_var_re {};
    double cplx_5_var_im {};
    double cplx_5_cov_re_im {};

    // double vector data (4 samples of size 3)
    std::vector<Eigen::Vector3d> vec_d_data;
    int vec_d_n {};
    Eigen::Vector3d vec_d_mean;
    Eigen::Vector3d vec_d_var;
    Eigen::Matrix3d vec_d_cov;

    // complex vector data (3 samples of size 2)
    std::vector<Eigen::Vector2cd> vec_c_data;
    int vec_c_n {};
    Eigen::Vector2cd vec_c_mean;
    Eigen::Vector2d vec_c_var_re;
    Eigen::Vector2d vec_c_var_im;
    Eigen::Vector2d vec_c_cov_re_im_diag;
    Eigen::Matrix2d vec_c_cov_rr;
    Eigen::Matrix2d vec_c_cov_ii;
    Eigen::Matrix2d vec_c_cov_ri;
};

#endif // SIMPLEMC_TESTS_ACCS_FIXTURE_HPP
