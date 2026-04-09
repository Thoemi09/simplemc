#include "./accs_fixture.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;

} // namespace

// Test accumulator concepts.
TEST_F(SimplemcAccs, VarAccConcepts) {
    // var_acc with scalar types
    static_assert(variance_accumulator<var_acc<double>>);
    static_assert(variance_accumulator<var_acc<std::complex<double>>>);

    // var_acc with static vector types
    static_assert(variance_accumulator<var_acc_static<double, 3>>);
    static_assert(variance_accumulator<var_acc_static<std::complex<double>, 3>>);

    // var_acc with dynamic vector types
    static_assert(variance_accumulator<var_acc_dynamic<double>>);
    static_assert(variance_accumulator<var_acc_dynamic<std::complex<double>>>);

    // multivalue_acc wrapping var_acc
    static_assert(basic_accumulator<multivalue_acc<var_acc<double>>>);
    static_assert(basic_accumulator<multivalue_acc<var_acc<std::complex<double>>>>);
    static_assert(basic_accumulator<multivalue_acc<var_acc_static<double, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<var_acc_static<std::complex<double>, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<var_acc_dynamic<double>>>);
    static_assert(basic_accumulator<multivalue_acc<var_acc_dynamic<std::complex<double>>>>);

    // block_acc wrapping var_acc
    static_assert(variance_accumulator<block_acc<var_acc<double>>>);
    static_assert(variance_accumulator<block_acc<var_acc<std::complex<double>>>>);
    static_assert(variance_accumulator<block_acc<var_acc_static<double, 3>>>);
    static_assert(variance_accumulator<block_acc<var_acc_static<std::complex<double>, 3>>>);
    static_assert(variance_accumulator<block_acc<var_acc_dynamic<double>>>);
    static_assert(variance_accumulator<block_acc<var_acc_dynamic<std::complex<double>>>>);

    // autocorr_acc wrapping var_acc
    static_assert(variance_accumulator<autocorr_acc<var_acc<double>>>);
    static_assert(variance_accumulator<autocorr_acc<var_acc<std::complex<double>>>>);
    static_assert(variance_accumulator<autocorr_acc<var_acc_static<double, 3>>>);
    static_assert(variance_accumulator<autocorr_acc<var_acc_static<std::complex<double>, 3>>>);
    static_assert(variance_accumulator<autocorr_acc<var_acc_dynamic<double>>>);
    static_assert(variance_accumulator<autocorr_acc<var_acc_dynamic<std::complex<double>>>>);
}

// Test empty accumulators.
TEST_F(SimplemcAccs, VarAccEmpty) {
    var_acc<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_acc_empty(acc_sd);
    acc_sd << acc_sd;
    check_acc_empty(acc_sd);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    var_acc<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_acc_empty(acc_sc);
    acc_sc << acc_sc;
    check_acc_empty(acc_sc);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    var_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_acc_empty(acc_st_d);
    acc_st_d << acc_st_d;
    check_acc_empty(acc_st_d);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    var_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_acc_empty(acc_st_c);
    acc_st_c << acc_st_c;
    check_acc_empty(acc_st_c);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    var_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_acc_empty(acc_dyn_d);
    acc_dyn_d << acc_dyn_d;
    check_acc_empty(acc_dyn_d);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    var_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_acc_empty(acc_dyn_c);
    acc_dyn_c << acc_dyn_c;
    check_acc_empty(acc_dyn_c);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Test scalar double accumulation.
TEST_F(SimplemcAccs, VarAccScalarDouble) {
    var_acc<double, standard> acc_std;
    var_acc<double, welford> acc_wel;
    for (const auto& x : dbl_5) {
        acc_std << x;
        acc_wel << x;
    }

    ASSERT_EQ(acc_std.count(), dbl_5_n);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);

    // mean
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);

    // variance of data
    check_near(acc_std.variance_of_data(), dbl_5_var_data, 1e-14);
    check_near(acc_wel.variance_of_data(), dbl_5_var_data, 1e-14);

    // variance (of the mean = var_data / N)
    check_near(acc_std.variance(), dbl_5_var_mean, 1e-14);
    check_near(acc_wel.variance(), dbl_5_var_mean, 1e-14);
}

// Test scalar complex accumulation.
TEST_F(SimplemcAccs, VarAccScalarComplex) {
    var_acc<cplx, standard> acc_std;
    var_acc<cplx, welford> acc_wel;
    for (const auto& z : cplx_5) {
        acc_std << z;
        acc_wel << z;
    }

    ASSERT_EQ(acc_std.count(), cplx_5_n);
    ASSERT_EQ(acc_wel.count(), cplx_5_n);

    // mean
    check_near(acc_std.mean(), cplx_5_mean, 1e-14);
    check_near(acc_wel.mean(), cplx_5_mean, 1e-14);

    // variance of real/imaginary parts
    check_near(acc_std.variance_of_real_data(), cplx_5_var_re, 1e-14);
    check_near(acc_std.variance_of_imag_data(), cplx_5_var_im, 1e-14);
    check_near(acc_wel.variance_of_real_data(), cplx_5_var_re, 1e-14);
    check_near(acc_wel.variance_of_imag_data(), cplx_5_var_im, 1e-14);

    // cross-covariance of real and imaginary parts (diagonal)
    check_near(acc_std.covariance_of_real_and_imag_data(), cplx_5_cov_re_im, 1e-14);
    check_near(acc_wel.covariance_of_real_and_imag_data(), cplx_5_cov_re_im, 1e-14);

    // total variance of data = var_re + var_im
    const auto expect_var_data = cplx_5_var_re + cplx_5_var_im;
    check_near(acc_std.variance_of_data(), expect_var_data, 1e-14);
    check_near(acc_wel.variance_of_data(), expect_var_data, 1e-14);

    // variance of the mean = (var_re + var_im) / N
    const auto expect_var = expect_var_data / static_cast<double>(cplx_5_n);
    check_near(acc_std.variance(), expect_var, 1e-14);
    check_near(acc_wel.variance(), expect_var, 1e-14);
}

// Test static vector double accumulation.
TEST_F(SimplemcAccs, VarAccStaticVectorDouble) {
    var_acc_static<double, 3, standard> acc_std;
    var_acc_static<double, 3, welford> acc_wel;
    for (const auto& v : vec_d_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_d_n);
    ASSERT_EQ(acc_wel.count(), vec_d_n);

    // mean
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);

    // variance of data
    check_near(acc_std.variance_of_data(), vec_d_var, 1e-14);
    check_near(acc_wel.variance_of_data(), vec_d_var, 1e-14);

    // variance of the mean = var / N
    Eigen::Vector3d expect_var_mean = vec_d_var / static_cast<double>(vec_d_n);
    check_near(acc_std.variance(), expect_var_mean, 1e-14);
    check_near(acc_wel.variance(), expect_var_mean, 1e-14);
}

// Test static vector complex accumulation.
TEST_F(SimplemcAccs, VarAccStaticVectorComplex) {
    var_acc_static<cplx, 2, standard> acc_std;
    var_acc_static<cplx, 2, welford> acc_wel;
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);

    // mean
    check_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_near(acc_wel.mean(), vec_c_mean, 1e-14);

    // variance of real/imaginary parts
    check_near(acc_std.variance_of_real_data(), vec_c_var_re, 1e-14);
    check_near(acc_std.variance_of_imag_data(), vec_c_var_im, 1e-14);
    check_near(acc_wel.variance_of_real_data(), vec_c_var_re, 1e-14);
    check_near(acc_wel.variance_of_imag_data(), vec_c_var_im, 1e-14);

    // cross-covariance diagonal
    check_near(acc_std.covariance_of_real_and_imag_data(), vec_c_cov_re_im_diag, 1e-14);
    check_near(acc_wel.covariance_of_real_and_imag_data(), vec_c_cov_re_im_diag, 1e-14);

    // total variance of data = var_re + var_im (component-wise)
    Eigen::Vector2d expect_var_data = vec_c_var_re + vec_c_var_im;
    check_near(acc_std.variance_of_data(), expect_var_data, 1e-14);
    check_near(acc_wel.variance_of_data(), expect_var_data, 1e-14);

    // variance of the mean = (var_re + var_im) / N
    Eigen::Vector2d expect_var = expect_var_data / static_cast<double>(vec_c_n);
    check_near(acc_std.variance(), expect_var, 1e-14);
    check_near(acc_wel.variance(), expect_var, 1e-14);
}

// Test dynamic vector double accumulation.
TEST_F(SimplemcAccs, VarAccDynamicVectorDouble) {
    var_acc_dynamic<double, standard> acc_std(3);
    var_acc_dynamic<double, welford> acc_wel(3);
    for (const auto& v : vec_d_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_d_n);
    ASSERT_EQ(acc_wel.count(), vec_d_n);

    // mean
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);

    // variance of data
    check_near(acc_std.variance_of_data(), vec_d_var, 1e-14);
    check_near(acc_wel.variance_of_data(), vec_d_var, 1e-14);

    // variance of the mean = var / N
    Eigen::Vector3d expect_var_mean = vec_d_var / static_cast<double>(vec_d_n);
    check_near(acc_std.variance(), expect_var_mean, 1e-14);
    check_near(acc_wel.variance(), expect_var_mean, 1e-14);
}

// Test dynamic vector complex accumulation.
TEST_F(SimplemcAccs, VarAccDynamicVectorComplex) {
    var_acc_dynamic<cplx, standard> acc_std(2);
    var_acc_dynamic<cplx, welford> acc_wel(2);
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);

    // mean
    check_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_near(acc_wel.mean(), vec_c_mean, 1e-14);

    // variance of real/imaginary parts
    check_near(acc_std.variance_of_real_data(), vec_c_var_re, 1e-14);
    check_near(acc_std.variance_of_imag_data(), vec_c_var_im, 1e-14);
    check_near(acc_wel.variance_of_real_data(), vec_c_var_re, 1e-14);
    check_near(acc_wel.variance_of_imag_data(), vec_c_var_im, 1e-14);

    // cross-covariance diagonal
    check_near(acc_std.covariance_of_real_and_imag_data(), vec_c_cov_re_im_diag, 1e-14);
    check_near(acc_wel.covariance_of_real_and_imag_data(), vec_c_cov_re_im_diag, 1e-14);

    // total variance of data = var_re + var_im (component-wise)
    Eigen::Vector2d expect_var_data = vec_c_var_re + vec_c_var_im;
    check_near(acc_std.variance_of_data(), expect_var_data, 1e-14);
    check_near(acc_wel.variance_of_data(), expect_var_data, 1e-14);

    // variance of the mean = (var_re + var_im) / N
    Eigen::Vector2d expect_var = expect_var_data / static_cast<double>(vec_c_n);
    check_near(acc_std.variance(), expect_var, 1e-14);
    check_near(acc_wel.variance(), expect_var, 1e-14);
}

// Test merging two var_acc accumulators.
TEST_F(SimplemcAccs, VarAccMergeScalar) {
    // split dbl_5 into two halves, merge, verify
    var_acc<double, standard> acc1_std, acc2_std;
    var_acc<double, welford> acc1_wel, acc2_wel;
    for (int i = 0; i < 3; ++i) {
        acc1_std << dbl_5[i];
        acc1_wel << dbl_5[i];
    }
    for (int i = 3; i < 5; ++i) {
        acc2_std << dbl_5[i];
        acc2_wel << dbl_5[i];
    }
    ASSERT_EQ(acc1_std.count(), 3);
    ASSERT_EQ(acc2_std.count(), 2);
    ASSERT_EQ(acc1_wel.count(), 3);
    ASSERT_EQ(acc2_wel.count(), 2);

    acc1_std << acc2_std;
    acc1_wel << acc2_wel;
    ASSERT_EQ(acc1_std.count(), dbl_5_n);
    ASSERT_EQ(acc2_std.count(), 2);
    ASSERT_EQ(acc1_wel.count(), dbl_5_n);
    ASSERT_EQ(acc2_wel.count(), 2);
    check_near(acc1_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc1_wel.mean(), dbl_5_mean, 1e-14);
    check_near(acc1_std.variance_of_data(), dbl_5_var_data, 1e-14);
    check_near(acc1_wel.variance_of_data(), dbl_5_var_data, 1e-14);
}

TEST_F(SimplemcAccs, VarAccMergeVector) {
    // split vec_d_data into two halves, merge, verify
    var_acc_static<double, 3, standard> acc1_std, acc2_std;
    var_acc_dynamic<double, welford> acc1_wel(3), acc2_wel(3);
    for (int i = 0; i < 2; ++i) {
        acc1_std << vec_d_data[i];
        acc1_wel << vec_d_data[i];
        acc2_std << vec_d_data[i + 2];
        acc2_wel << vec_d_data[i + 2];
    }
    ASSERT_EQ(acc1_std.count(), 2);
    ASSERT_EQ(acc2_std.count(), 2);
    ASSERT_EQ(acc1_wel.count(), 2);
    ASSERT_EQ(acc2_wel.count(), 2);

    acc1_std << acc2_std;
    acc1_wel << acc2_wel;
    ASSERT_EQ(acc1_std.count(), vec_d_n);
    ASSERT_EQ(acc2_std.count(), 2);
    ASSERT_EQ(acc1_wel.count(), vec_d_n);
    ASSERT_EQ(acc2_wel.count(), 2);
    check_near(acc1_std.mean(), vec_d_mean, 1e-14);
    check_near(acc1_wel.mean(), vec_d_mean, 1e-14);
    check_near(acc1_std.variance_of_data(), vec_d_var, 1e-14);
    check_near(acc1_wel.variance_of_data(), vec_d_var, 1e-14);
}

// Test reseting an accumulator.
TEST_F(SimplemcAccs, VarAccReset) {
    var_acc<double> acc;
    for (const auto& x : dbl_5) {
        acc << x;
    }
    ASSERT_FALSE(acc.empty());

    acc.reset();
    check_acc_empty(acc);
}

// Test constructing a var_acc from data and count.
TEST_F(SimplemcAccs, VarAccDataConstructorDouble) {
    // scalar data constructor (valid case)
    var_acc<double> acc;
    for (const auto& x : dbl_5) {
        acc << x;
    }
    var_acc<double> acc_copy(acc.mdata(), acc.cdata(), acc.count());
    ASSERT_EQ(acc_copy.count(), acc.count());
    ASSERT_EQ(acc_copy.mdata(), acc.mdata());
    ASSERT_EQ(acc_copy.cdata(), acc.cdata());
    check_near(acc_copy.mean(), dbl_5_mean, 1e-14);
    check_near(acc_copy.variance_of_data(), dbl_5_var_data, 1e-14);

    // dynamic double data constructor
    var_acc_dynamic<double> acc_dyn(3);
    for (const auto& v : vec_d_data) {
        acc_dyn << v;
    }
    var_acc_dynamic<double> acc_dyn_copy(acc_dyn.mdata(), acc_dyn.cdata(), acc_dyn.count());
    ASSERT_EQ(acc_dyn_copy.count(), acc_dyn.count());
    ASSERT_TRUE(acc_dyn_copy.mdata() == acc_dyn.mdata());
    ASSERT_TRUE(acc_dyn_copy.cdata() == acc_dyn.cdata());
    check_near(acc_dyn_copy.mean(), vec_d_mean, 1e-14);
    check_near(acc_dyn_copy.variance_of_data(), vec_d_var, 1e-14);

    // dynamic complex data constructor
    var_acc_dynamic<cplx> acc_dyn_c(2);
    for (const auto& v : vec_c_data) {
        acc_dyn_c << v;
    }
    var_acc_dynamic<cplx> acc_dyn_c_copy(
        acc_dyn_c.mdata(), acc_dyn_c.rdata(), acc_dyn_c.idata(), acc_dyn_c.cdata(), acc_dyn_c.count());
    ASSERT_EQ(acc_dyn_c_copy.count(), acc_dyn_c.count());
    ASSERT_TRUE(acc_dyn_c_copy.mdata() == acc_dyn_c.mdata());
    ASSERT_TRUE(acc_dyn_c_copy.rdata() == acc_dyn_c.rdata());
    ASSERT_TRUE(acc_dyn_c_copy.idata() == acc_dyn_c.idata());
    ASSERT_TRUE(acc_dyn_c_copy.cdata() == acc_dyn_c.cdata());
    check_near(acc_dyn_c_copy.mean(), vec_c_mean, 1e-14);
    check_near(acc_dyn_c_copy.variance_of_real_data(), vec_c_var_re, 1e-14);
    check_near(acc_dyn_c_copy.variance_of_imag_data(), vec_c_var_im, 1e-14);
}

TEST_F(SimplemcAccs, VarAccDataConstructorComplex) {
    // scalar complex data constructor
    var_acc<cplx> acc_c;
    for (const auto& z : cplx_5) {
        acc_c << z;
    }
    var_acc<cplx> acc_c_copy(acc_c.mdata(), acc_c.rdata(), acc_c.idata(), acc_c.cdata(), acc_c.count());
    ASSERT_EQ(acc_c_copy.count(), acc_c.count());
    ASSERT_EQ(acc_c_copy.mdata(), acc_c.mdata());
    ASSERT_EQ(acc_c_copy.rdata(), acc_c.rdata());
    ASSERT_EQ(acc_c_copy.idata(), acc_c.idata());
    ASSERT_EQ(acc_c_copy.cdata(), acc_c.cdata());
    check_near(acc_c_copy.mean(), cplx_5_mean, 1e-14);
    check_near(acc_c_copy.variance_of_real_data(), cplx_5_var_re, 1e-14);
    check_near(acc_c_copy.variance_of_imag_data(), cplx_5_var_im, 1e-14);

    // dynamic complex data constructor
    var_acc_dynamic<cplx> acc_dyn_c(2);
    for (const auto& v : vec_c_data) {
        acc_dyn_c << v;
    }
    var_acc_dynamic<cplx> acc_dyn_c_copy(
        acc_dyn_c.mdata(), acc_dyn_c.rdata(), acc_dyn_c.idata(), acc_dyn_c.cdata(), acc_dyn_c.count());
    ASSERT_EQ(acc_dyn_c_copy.count(), acc_dyn_c.count());
    ASSERT_TRUE(acc_dyn_c_copy.mdata() == acc_dyn_c.mdata());
    ASSERT_TRUE(acc_dyn_c_copy.rdata() == acc_dyn_c.rdata());
    ASSERT_TRUE(acc_dyn_c_copy.idata() == acc_dyn_c.idata());
    ASSERT_TRUE(acc_dyn_c_copy.cdata() == acc_dyn_c.cdata());
    check_near(acc_dyn_c_copy.mean(), vec_c_mean, 1e-14);
    check_near(acc_dyn_c_copy.variance_of_real_data(), vec_c_var_re, 1e-14);
    check_near(acc_dyn_c_copy.variance_of_imag_data(), vec_c_var_im, 1e-14);
}

// Test that constructing dynamic var_acc with invalid sizes throws.
// Note: only m == 0 is tested here; m < 0 would trigger an Eigen internal assertion
// (in the member initializer) before our check runs.
TEST_F(SimplemcAccs, VarAccDynamicConstructorInvalid) {
    EXPECT_THROW(var_acc_dynamic<double>(0), simplemc_exception);

    // data constructor: zero-size vec_type triggers the same exception
    Eigen::VectorXd empty_vec(0);
    EXPECT_THROW((var_acc_dynamic<double>(empty_vec, empty_vec, 0)), simplemc_exception);
}

// Test factory function for var_acc.
TEST_F(SimplemcAccs, VarAccFactoryScalar) {
    // default (welford) varalg
    auto acc_wel = make_var_acc(dbl_5);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);
    check_near(acc_wel.variance_of_data(), dbl_5_var_data, 1e-14);

    // explicit standard varalg
    auto acc_std = make_var_acc<standard>(dbl_5);
    ASSERT_EQ(acc_std.count(), dbl_5_n);
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc_std.variance_of_data(), dbl_5_var_data, 1e-14);

    // with a non-null shift t
    const double shift = 1.0;
    auto acc_shifted = make_var_acc(dbl_5, std::optional<double>(shift));
    ASSERT_EQ(acc_shifted.count(), dbl_5_n);
    check_near(acc_shifted.mean() + shift, dbl_5_mean, 1e-14);
}

TEST_F(SimplemcAccs, VarAccFactoryVector) {
    // default (welford) varalg
    auto acc_wel = make_var_acc(vec_d_data);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.variance_of_data(), vec_d_var, 1e-14);

    // explicit standard varalg
    auto acc_std = make_var_acc<standard>(vec_d_data);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_std.variance_of_data(), vec_d_var, 1e-14);

    // with a non-null vector shift t
    const Eigen::Vector3d shift(1.0, 1.0, 1.0);
    auto acc_shifted = make_var_acc(vec_d_data, std::optional<Eigen::Vector3d>(shift));
    ASSERT_EQ(acc_shifted.count(), vec_d_n);
    const Eigen::Vector3d shifted_mean = (acc_shifted.mean() + shift).eval();
    check_near(shifted_mean, vec_d_mean, 1e-14);
}

// Test that make_var_acc throws on an empty range (detail::make_acc exception path).
TEST_F(SimplemcAccs, VarAccFactoryEmptyRange) {
    std::vector<double> empty;
    EXPECT_THROW((void)make_var_acc(empty), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_vec;
    EXPECT_THROW((void)make_var_acc(empty_vec), simplemc_exception);
}

// Test block_acc wrapping var_acc.
TEST_F(SimplemcAccs, VarAccBlockScalar) {
    // dbl_5 = {1,2,3,4,5} with block_size = 2:
    //   Block 1: mean(1,2) = 1.5
    //   Block 2: mean(3,4) = 3.5
    //   Sample 5 in incomplete block (not flushed).
    // Inner acc gets {1.5, 3.5}:
    //   mean = 2.5, var_data = 2.0, var_mean = 1.0
    block_acc<var_acc<double>> bacc(2);
    for (const auto& x : dbl_5) {
        bacc << x;
    }

    ASSERT_EQ(bacc.block_size(), 2UL);
    ASSERT_EQ(bacc.count(), 2UL);
    ASSERT_EQ(bacc.total_count(), 5UL);
    ASSERT_FALSE(bacc.empty());

    check_near(bacc.mean(), 2.5, 1e-14);
    check_near(bacc.accumulator().variance_of_data(), 2.0, 1e-14);
    check_near(bacc.variance(), 1.0, 1e-14);
}

TEST_F(SimplemcAccs, VarAccBlockEmpty) {
    block_acc<var_acc<double>> bacc(3);
    ASSERT_TRUE(bacc.empty());
    ASSERT_EQ(bacc.count(), 0UL);
    ASSERT_EQ(bacc.block_size(), 3UL);
}

TEST_F(SimplemcAccs, VarAccBlockFactory) {
    auto bacc = make_block_var_acc(dbl_5, 2);
    ASSERT_EQ(bacc.count(), 2UL);
    check_near(bacc.mean(), 2.5, 1e-14);
}

// Test autocorr_acc wrapping var_acc.
TEST_F(SimplemcAccs, VarAccAutocorrStructural) {
    // Feed dbl_5 into autocorr_acc with factor 2.
    autocorr_acc<var_acc<double>> aacc(1, 2);
    for (const auto& x : dbl_5) {
        aacc << x;
    }

    ASSERT_FALSE(aacc.empty());
    ASSERT_EQ(aacc.factor(), 2UL);
    ASSERT_GE(aacc.num_levels(), 2UL);

    // Level 0 gets all samples directly.
    ASSERT_EQ(aacc.count(0), dbl_5_n);
    check_near(aacc.mean(), dbl_5_mean, 1e-14);

    // Higher levels have fewer completed blocks.
    ASSERT_LE(aacc.count(1), aacc.count(0));
}

TEST_F(SimplemcAccs, VarAccAutocorrFactory) {
    auto aacc = make_autocorr_var_acc(dbl_5);
    ASSERT_EQ(aacc.count(0), dbl_5_n);
    check_near(aacc.mean(), dbl_5_mean, 1e-14);
}

// Test index-based accumulation for var_acc.
TEST_F(SimplemcAccs, VarAccIndexBased) {
    var_acc_static<double, 3> acc;
    for (const auto& v : vec_d_data) {
        acc[1] << v[1];
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    check_near(acc.mean()[1], vec_d_mean[1], 1e-14);
    check_near(acc.variance_of_data()[1], vec_d_var[1], 1e-14);
}

TEST_F(SimplemcAccs, VarAccAccumulateWithIndices) {
    // use accumulate(range, indices) to accumulate only element 0 and 2 of 3-element accumulator
    var_acc_static<double, 3> acc_wel;
    var_acc_static<double, 3, standard> acc_std;
    std::vector<long> idxs = { 0, 2 };
    for (const auto& v : vec_d_data) {
        Eigen::Vector2d sub(v[0], v[2]);
        acc_wel.accumulate(sub, idxs);
        acc_std.accumulate(sub, idxs);
    }

    // only elements 0 and 2 should have the correct mean/variance, element 1 is 0
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    Eigen::Vector3d expected_mean = { vec_d_mean[0], 0.0, vec_d_mean[2] };
    Eigen::Vector3d expected_var = { vec_d_var[0], 0.0, vec_d_var[2] };
    check_near(acc_wel.mean(), expected_mean, 1e-14);
    check_near(acc_std.mean(), expected_mean, 1e-14);
    check_near(acc_wel.variance_of_data(), expected_var, 1e-14);
    check_near(acc_std.variance_of_data(), expected_var, 1e-14);
}

TEST_F(SimplemcAccs, VarAccAccumulateConsecutive) {
    // use accumulate(range, start_index) to accumulate elements 1 and 2 of 3-element accumulator
    var_acc_static<double, 3> acc_wel;
    var_acc_static<double, 3, standard> acc_std;
    for (const auto& v : vec_d_data) {
        std::vector<double> sub = { v[1], v[2] };
        acc_wel.accumulate(sub, 1);
        acc_std.accumulate(sub, 1);
    }

    // only elements 1 and 2 should have the correct mean/variance, element 0 is 0
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    Eigen::Vector3d expected_mean = { 0.0, vec_d_mean[1], vec_d_mean[2] };
    Eigen::Vector3d expected_var = { 0.0, vec_d_var[1], vec_d_var[2] };
    check_near(acc_wel.mean(), expected_mean, 1e-14);
    check_near(acc_std.mean(), expected_mean, 1e-14);
    check_near(acc_wel.variance_of_data(), expected_var, 1e-14);
    check_near(acc_std.variance_of_data(), expected_var, 1e-14);
}

// Test multivalue_acc wrapper for var_acc.
TEST_F(SimplemcAccs, VarAccMultivalue) {
    var_acc_static<double, 3> acc_wel;
    var_acc_static<double, 3, standard> acc_std;
    auto mva_wel = acc_wel.make_mva();
    auto mva_std = acc_std.make_mva();

    // accumulate element-by-element for each sample
    for (const auto& v : vec_d_data) {
        for (int i = 0; i < 3; ++i) {
            mva_wel[i] << v[i];
            mva_std[i] << v[i];
        }
        mva_wel.increment_count();
        mva_std.increment_count();
    }

    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.variance_of_data(), vec_d_var, 1e-14);
    check_near(acc_std.variance_of_data(), vec_d_var, 1e-14);
}

// Test multivalue_acc vector stream operator (operator<<(W v)).
TEST_F(SimplemcAccs, VarAccMultivalueVectorStream) {
    var_acc_static<double, 3> acc;
    auto mva = acc.make_mva();

    for (const auto& v : vec_d_data) {
        mva << v;
        mva.increment_count();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    check_near(acc.mean(), vec_d_mean, 1e-14);
    check_near(acc.variance_of_data(), vec_d_var, 1e-14);
}

// Test multivalue_acc accumulate with consecutive indices.
TEST_F(SimplemcAccs, VarAccMultivalueAccumulateConsecutive) {
    var_acc_static<double, 3> acc;
    auto mva = acc.make_mva();

    for (const auto& v : vec_d_data) {
        std::vector<double> sub = { v[1], v[2] };
        mva.accumulate(sub, 1);
        mva.increment_count();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    Eigen::Vector3d expected = { 0.0, vec_d_mean[1], vec_d_mean[2] };
    check_near(acc.mean(), expected, 1e-14);
}

// Test multivalue_acc accumulate with arbitrary indices.
TEST_F(SimplemcAccs, VarAccMultivalueAccumulateWithIndices) {
    var_acc_static<double, 3> acc;
    auto mva = acc.make_mva();
    std::vector<long> idxs = { 0, 2 };

    for (const auto& v : vec_d_data) {
        Eigen::Vector2d sub(v[0], v[2]);
        mva.accumulate(sub, idxs);
        mva.increment_count();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    Eigen::Vector3d expected = { vec_d_mean[0], 0.0, vec_d_mean[2] };
    check_near(acc.mean(), expected, 1e-14);
}

// Test multivalue_acc accessors: size(), count(), empty(), accumulator().
TEST_F(SimplemcAccs, VarAccMultivalueAccessors) {
    var_acc_static<double, 3> acc;
    auto mva = acc.make_mva();

    ASSERT_EQ(mva.size(), 3);
    ASSERT_EQ(mva.count(), 0);
    ASSERT_TRUE(mva.empty());
    ASSERT_EQ(&mva.accumulator(), &acc);

    // const accessor
    const auto& mva_const = mva;
    ASSERT_EQ(&mva_const.accumulator(), &acc);

    mva[0] << 1.0;
    mva.increment_count();
    ASSERT_EQ(mva.count(), 1);
    ASSERT_FALSE(mva.empty());

    // increment_count with inc > 1
    mva[0] << 2.0;
    mva[0] << 3.0;
    mva.increment_count(2);
    ASSERT_EQ(mva.count(), 3);
}
