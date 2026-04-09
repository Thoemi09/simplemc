#include "./accs_fixture.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
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
TEST_F(SimplemcAccs, CovarAccConcepts) {
    // covar_acc with scalar types
    static_assert(covariance_accumulator<covar_acc<double>>);
    static_assert(covariance_accumulator<covar_acc<std::complex<double>>>);

    // covar_acc with static vector types
    static_assert(covariance_accumulator<covar_acc_static<double, 3>>);
    static_assert(covariance_accumulator<covar_acc_static<std::complex<double>, 3>>);

    // covar_acc with dynamic vector types
    static_assert(covariance_accumulator<covar_acc_dynamic<double>>);
    static_assert(covariance_accumulator<covar_acc_dynamic<std::complex<double>>>);

    // multivalue_acc wrapping covar_acc
    static_assert(basic_accumulator<multivalue_acc<covar_acc<double>>>);
    static_assert(basic_accumulator<multivalue_acc<covar_acc<std::complex<double>>>>);
    static_assert(basic_accumulator<multivalue_acc<covar_acc_static<double, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<covar_acc_static<std::complex<double>, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<covar_acc_dynamic<double>>>);
    static_assert(basic_accumulator<multivalue_acc<covar_acc_dynamic<std::complex<double>>>>);

    // block_acc wrapping covar_acc
    static_assert(covariance_accumulator<block_acc<covar_acc<double>>>);
    static_assert(covariance_accumulator<block_acc<covar_acc<std::complex<double>>>>);
    static_assert(covariance_accumulator<block_acc<covar_acc_static<double, 3>>>);
    static_assert(covariance_accumulator<block_acc<covar_acc_static<std::complex<double>, 3>>>);
    static_assert(covariance_accumulator<block_acc<covar_acc_dynamic<double>>>);
    static_assert(covariance_accumulator<block_acc<covar_acc_dynamic<std::complex<double>>>>);

    // autocorr_acc wrapping covar_acc
    static_assert(covariance_accumulator<autocorr_acc<covar_acc<double>>>);
    static_assert(covariance_accumulator<autocorr_acc<covar_acc<std::complex<double>>>>);
    static_assert(covariance_accumulator<autocorr_acc<covar_acc_static<double, 3>>>);
    static_assert(covariance_accumulator<autocorr_acc<covar_acc_static<std::complex<double>, 3>>>);
    static_assert(covariance_accumulator<autocorr_acc<covar_acc_dynamic<double>>>);
    static_assert(covariance_accumulator<autocorr_acc<covar_acc_dynamic<std::complex<double>>>>);
}

// Test empty accumulators.
TEST_F(SimplemcAccs, CovarAccEmpty) {
    covar_acc<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_acc_empty(acc_sd);
    acc_sd << acc_sd;
    check_acc_empty(acc_sd);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    covar_acc<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_acc_empty(acc_sc);
    acc_sc << acc_sc;
    check_acc_empty(acc_sc);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    covar_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_acc_empty(acc_st_d);
    acc_st_d << acc_st_d;
    check_acc_empty(acc_st_d);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    covar_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_acc_empty(acc_st_c);
    acc_st_c << acc_st_c;
    check_acc_empty(acc_st_c);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    covar_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_acc_empty(acc_dyn_d);
    acc_dyn_d << acc_dyn_d;
    check_acc_empty(acc_dyn_d);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    covar_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_acc_empty(acc_dyn_c);
    acc_dyn_c << acc_dyn_c;
    check_acc_empty(acc_dyn_c);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Test scalar double accumulation.
TEST_F(SimplemcAccs, CovarAccScalarDouble) {
    covar_acc<double, standard> acc_std;
    covar_acc<double, welford> acc_wel;
    for (const auto& x : dbl_5) {
        acc_std << x;
        acc_wel << x;
    }

    ASSERT_EQ(acc_std.count(), dbl_5_n);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    ASSERT_FALSE(acc_std.empty());
    ASSERT_FALSE(acc_wel.empty());

    // mean
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);

    // covariance of data (= sample variance for scalar)
    check_near(acc_std.covariance_of_data(), dbl_5_var_data, 1e-14);
    check_near(acc_wel.covariance_of_data(), dbl_5_var_data, 1e-14);

    // covariance of the mean
    check_near(acc_std.covariance(), dbl_5_var_mean, 1e-14);
    check_near(acc_wel.covariance(), dbl_5_var_mean, 1e-14);

    // variance (diagonal of covariance = covariance for scalar)
    check_near(acc_std.variance(), dbl_5_var_mean, 1e-14);
    check_near(acc_wel.variance(), dbl_5_var_mean, 1e-14);
}

// Test scalar complex accumulation.
TEST_F(SimplemcAccs, CovarAccScalarComplex) {
    covar_acc<cplx, standard> acc_std;
    covar_acc<cplx, welford> acc_wel;
    for (const auto& z : cplx_5) {
        acc_std << z;
        acc_wel << z;
    }

    ASSERT_EQ(acc_std.count(), cplx_5_n);
    ASSERT_EQ(acc_wel.count(), cplx_5_n);

    // mean
    check_near(acc_std.mean(), cplx_5_mean, 1e-14);
    check_near(acc_wel.mean(), cplx_5_mean, 1e-14);

    // covariance components
    check_near(acc_std.covariance_of_real_data(), cplx_5_var_re, 1e-14);
    check_near(acc_std.covariance_of_imag_data(), cplx_5_var_im, 1e-14);
    check_near(acc_std.covariance_of_real_and_imag_data(), cplx_5_cov_re_im, 1e-14);
    check_near(acc_wel.covariance_of_real_data(), cplx_5_var_re, 1e-14);
    check_near(acc_wel.covariance_of_imag_data(), cplx_5_var_im, 1e-14);
    check_near(acc_wel.covariance_of_real_and_imag_data(), cplx_5_cov_re_im, 1e-14);

    // total covariance of data = var_re + var_im
    const auto expect_cov_data = cplx_5_var_re + cplx_5_var_im;
    check_near(acc_std.covariance_of_data(), expect_cov_data, 1e-14);
    check_near(acc_wel.covariance_of_data(), expect_cov_data, 1e-14);
}

// Test static vector double accumulation.
TEST_F(SimplemcAccs, CovarAccStaticVectorDouble) {
    covar_acc_static<double, 3, standard> acc_std;
    covar_acc_static<double, 3, welford> acc_wel;
    for (const auto& v : vec_d_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_d_n);
    ASSERT_EQ(acc_wel.count(), vec_d_n);

    // mean
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);

    // full covariance of data (3x3 matrix)
    check_near(acc_std.covariance_of_data(), vec_d_cov, 1e-14);
    check_near(acc_wel.covariance_of_data(), vec_d_cov, 1e-14);

    // covariance of the mean = cov_data / N
    Eigen::Matrix3d expect_cov_mean = vec_d_cov / static_cast<double>(vec_d_n);
    check_near(acc_std.covariance(), expect_cov_mean, 1e-14);
    check_near(acc_wel.covariance(), expect_cov_mean, 1e-14);

    // variance = diagonal of covariance (vector)
    Eigen::Vector3d expect_var = vec_d_var / static_cast<double>(vec_d_n);
    check_near(acc_std.variance(), expect_var, 1e-14);
    check_near(acc_wel.variance(), expect_var, 1e-14);
}

// Test static vector complex accumulation.
TEST_F(SimplemcAccs, CovarAccStaticVectorComplex) {
    covar_acc_static<cplx, 2, standard> acc_std;
    covar_acc_static<cplx, 2, welford> acc_wel;
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);

    // mean
    check_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_near(acc_wel.mean(), vec_c_mean, 1e-14);

    // component covariance matrices
    check_near(acc_std.covariance_of_real_data(), vec_c_cov_rr, 1e-14);
    check_near(acc_std.covariance_of_imag_data(), vec_c_cov_ii, 1e-14);
    check_near(acc_std.covariance_of_real_and_imag_data(), vec_c_cov_ri, 1e-14);
    check_near(acc_wel.covariance_of_real_data(), vec_c_cov_rr, 1e-14);
    check_near(acc_wel.covariance_of_imag_data(), vec_c_cov_ii, 1e-14);
    check_near(acc_wel.covariance_of_real_and_imag_data(), vec_c_cov_ri, 1e-14);
}

// Test dynamic vector double accumulation.
TEST_F(SimplemcAccs, CovarAccDynamicVectorDouble) {
    covar_acc_dynamic<double, standard> acc_std(3);
    covar_acc_dynamic<double, welford> acc_wel(3);
    for (const auto& v : vec_d_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_d_n);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.size(), 3);

    // mean
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);

    // covariance of data
    check_near(acc_std.covariance_of_data(), vec_d_cov, 1e-14);
    check_near(acc_wel.covariance_of_data(), vec_d_cov, 1e-14);
}

// Test dynamic vector complex accumulation.
TEST_F(SimplemcAccs, CovarAccDynamicVectorComplex) {
    covar_acc_dynamic<cplx, standard> acc_std(2);
    covar_acc_dynamic<cplx, welford> acc_wel(2);
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);
    ASSERT_EQ(acc_std.size(), 2);

    // mean
    check_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_near(acc_wel.mean(), vec_c_mean, 1e-14);

    // component covariance matrices
    check_near(acc_std.covariance_of_real_data(), vec_c_cov_rr, 1e-14);
    check_near(acc_std.covariance_of_imag_data(), vec_c_cov_ii, 1e-14);
    check_near(acc_std.covariance_of_real_and_imag_data(), vec_c_cov_ri, 1e-14);
    check_near(acc_wel.covariance_of_real_data(), vec_c_cov_rr, 1e-14);
    check_near(acc_wel.covariance_of_imag_data(), vec_c_cov_ii, 1e-14);
    check_near(acc_wel.covariance_of_real_and_imag_data(), vec_c_cov_ri, 1e-14);
}

// Test merging two covar_acc accumulators.
TEST_F(SimplemcAccs, CovarAccMergeScalar) {
    // split dbl_5 into two halves, merge, verify
    covar_acc<double, standard> acc1_std, acc2_std;
    covar_acc<double, welford> acc1_wel, acc2_wel;
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
    check_near(acc1_std.covariance_of_data(), dbl_5_var_data, 1e-14);
    check_near(acc1_wel.covariance_of_data(), dbl_5_var_data, 1e-14);
}

TEST_F(SimplemcAccs, CovarAccMergeVector) {
    // split vec_d_data into two halves, merge, verify
    covar_acc_static<double, 3, standard> acc1_std, acc2_std;
    covar_acc_dynamic<double, welford> acc1_wel(3), acc2_wel(3);
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
    check_near(acc1_std.covariance_of_data(), vec_d_cov, 1e-14);
    check_near(acc1_wel.covariance_of_data(), vec_d_cov, 1e-14);
}

// Test reseting an accumulator.
TEST_F(SimplemcAccs, CovarAccReset) {
    covar_acc<double> acc;
    for (const auto& x : dbl_5) {
        acc << x;
    }
    ASSERT_FALSE(acc.empty());

    acc.reset();
    check_acc_empty(acc);
}

// Test constructing a covar_acc from data and count.
TEST_F(SimplemcAccs, CovarAccDataConstructor) {
    // scalar data constructor (valid case)
    covar_acc<double> acc;
    for (const auto& x : dbl_5) {
        acc << x;
    }
    covar_acc<double> acc_copy(acc.mdata(), acc.cdata(), acc.count());
    ASSERT_EQ(acc_copy.count(), acc.count());
    ASSERT_EQ(acc_copy.mdata(), acc.mdata());
    ASSERT_EQ(acc_copy.cdata(), acc.cdata());
    check_near(acc_copy.mean(), dbl_5_mean, 1e-14);
    check_near(acc_copy.covariance_of_data(), dbl_5_var_data, 1e-14);

    // scalar complex data constructor
    covar_acc<cplx> acc_c;
    for (const auto& z : cplx_5) {
        acc_c << z;
    }
    covar_acc<cplx> acc_c_copy(
        acc_c.mdata(), acc_c.rdata(), acc_c.idata(), acc_c.cdata(), acc_c.count());
    ASSERT_EQ(acc_c_copy.count(), acc_c.count());
    ASSERT_EQ(acc_c_copy.mdata(), acc_c.mdata());
    ASSERT_EQ(acc_c_copy.rdata(), acc_c.rdata());
    ASSERT_EQ(acc_c_copy.idata(), acc_c.idata());
    ASSERT_EQ(acc_c_copy.cdata(), acc_c.cdata());
    check_near(acc_c_copy.mean(), cplx_5_mean, 1e-14);
    check_near(acc_c_copy.covariance_of_real_data(), cplx_5_var_re, 1e-14);
    check_near(acc_c_copy.covariance_of_imag_data(), cplx_5_var_im, 1e-14);

    // dynamic double data constructor
    covar_acc_dynamic<double> acc_dyn(3);
    for (const auto& v : vec_d_data) {
        acc_dyn << v;
    }
    covar_acc_dynamic<double> acc_dyn_copy(acc_dyn.mdata(), acc_dyn.cdata(), acc_dyn.count());
    ASSERT_EQ(acc_dyn_copy.count(), acc_dyn.count());
    ASSERT_TRUE(acc_dyn_copy.mdata() == acc_dyn.mdata());
    ASSERT_TRUE(acc_dyn_copy.cdata() == acc_dyn.cdata());
    check_near(acc_dyn_copy.mean(), vec_d_mean, 1e-14);
    check_near(acc_dyn_copy.covariance_of_data(), vec_d_cov, 1e-14);

    // dynamic complex data constructor
    covar_acc_dynamic<cplx> acc_dyn_c(2);
    for (const auto& v : vec_c_data) {
        acc_dyn_c << v;
    }
    covar_acc_dynamic<cplx> acc_dyn_c_copy(
        acc_dyn_c.mdata(), acc_dyn_c.rdata(), acc_dyn_c.idata(), acc_dyn_c.cdata(), acc_dyn_c.count());
    ASSERT_EQ(acc_dyn_c_copy.count(), acc_dyn_c.count());
    ASSERT_TRUE(acc_dyn_c_copy.mdata() == acc_dyn_c.mdata());
    ASSERT_TRUE(acc_dyn_c_copy.rdata() == acc_dyn_c.rdata());
    ASSERT_TRUE(acc_dyn_c_copy.idata() == acc_dyn_c.idata());
    ASSERT_TRUE(acc_dyn_c_copy.cdata() == acc_dyn_c.cdata());
    check_near(acc_dyn_c_copy.mean(), vec_c_mean, 1e-14);
    check_near(acc_dyn_c_copy.covariance_of_real_data(), vec_c_cov_rr, 1e-14);
    check_near(acc_dyn_c_copy.covariance_of_imag_data(), vec_c_cov_ii, 1e-14);
    check_near(acc_dyn_c_copy.covariance_of_real_and_imag_data(), vec_c_cov_ri, 1e-14);
}

// Test that constructing dynamic covar_acc with invalid sizes throws.
// Note: only m == 0 is tested here; m < 0 would trigger an Eigen internal assertion
// (in the member initializer) before our check runs.
TEST_F(SimplemcAccs, CovarAccDynamicConstructorInvalid) {
    EXPECT_THROW(covar_acc_dynamic<double>(0), simplemc_exception);

    // data constructor: zero-size vec_type triggers the same exception
    Eigen::VectorXd empty_vec(0);
    Eigen::MatrixXd empty_mat(0, 0);
    EXPECT_THROW((covar_acc_dynamic<double>(empty_vec, empty_mat, 0)), simplemc_exception);
}

// Test factory function for covar_acc.
TEST_F(SimplemcAccs, CovarAccFactory) {
    // default (welford) varalg
    auto acc_wel = make_covar_acc(dbl_5);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);
    check_near(acc_wel.covariance_of_data(), dbl_5_var_data, 1e-14);

    // explicit standard varalg
    auto acc_std = make_covar_acc<standard>(dbl_5);
    ASSERT_EQ(acc_std.count(), dbl_5_n);
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc_std.covariance_of_data(), dbl_5_var_data, 1e-14);

    // with a non-null shift t
    const double shift = 1.0;
    auto acc_shifted = make_covar_acc(dbl_5, std::optional<double>(shift));
    ASSERT_EQ(acc_shifted.count(), dbl_5_n);
    check_near(acc_shifted.mean() + shift, dbl_5_mean, 1e-14);
}

TEST_F(SimplemcAccs, CovarAccFactoryVector) {
    // default (welford) varalg
    auto acc_wel = make_covar_acc(vec_d_data);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.covariance_of_data(), vec_d_cov, 1e-14);

    // explicit standard varalg
    auto acc_std = make_covar_acc<standard>(vec_d_data);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_std.covariance_of_data(), vec_d_cov, 1e-14);

    // with a non-null vector shift t
    const Eigen::Vector3d shift(1.0, 1.0, 1.0);
    auto acc_shifted = make_covar_acc(vec_d_data, std::optional<Eigen::Vector3d>(shift));
    ASSERT_EQ(acc_shifted.count(), vec_d_n);
    const Eigen::Vector3d shifted_mean = (acc_shifted.mean() + shift).eval();
    check_near(shifted_mean, vec_d_mean, 1e-14);
}

// Test that make_covar_acc throws on an empty range (detail::make_acc exception path).
TEST_F(SimplemcAccs, CovarAccFactoryEmptyRange) {
    std::vector<double> empty;
    EXPECT_THROW((void)make_covar_acc(empty), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_vec;
    EXPECT_THROW((void)make_covar_acc(empty_vec), simplemc_exception);
}

// Test block_acc wrapping covar_acc.
TEST_F(SimplemcAccs, CovarAccBlockScalar) {
    // dbl_5 = {1,2,3,4,5} with block_size = 2:
    //   Block 1: mean(1,2) = 1.5
    //   Block 2: mean(3,4) = 3.5
    //   Sample 5 in incomplete block (not flushed).
    // Inner acc gets {1.5, 3.5}:
    //   cov_data = var = 2.0
    block_acc<covar_acc<double>> bacc(2);
    for (const auto& x : dbl_5) {
        bacc << x;
    }

    ASSERT_EQ(bacc.block_size(), 2UL);
    ASSERT_EQ(bacc.count(), 2UL);
    ASSERT_EQ(bacc.total_count(), 5UL);
    ASSERT_FALSE(bacc.empty());

    check_near(bacc.mean(), 2.5, 1e-14);
    check_near(bacc.accumulator().covariance_of_data(), 2.0, 1e-14);
    check_near(bacc.covariance(), 1.0, 1e-14);
}

TEST_F(SimplemcAccs, CovarAccBlockEmpty) {
    block_acc<covar_acc<double>> bacc(3);
    ASSERT_TRUE(bacc.empty());
    ASSERT_EQ(bacc.count(), 0UL);
    ASSERT_EQ(bacc.block_size(), 3UL);
}

TEST_F(SimplemcAccs, CovarAccBlockFactory) {
    auto bacc = make_block_covar_acc(dbl_5, 2);
    ASSERT_EQ(bacc.count(), 2UL);
    check_near(bacc.mean(), 2.5, 1e-14);
}

// Test autocorr_acc wrapping covar_acc.
TEST_F(SimplemcAccs, CovarAccAutocorrStructural) {
    // Feed dbl_5 into autocorr_acc with factor 2.
    autocorr_acc<covar_acc<double>> aacc(1, 2);
    for (const auto& x : dbl_5) {
        aacc << x;
    }

    ASSERT_FALSE(aacc.empty());
    ASSERT_EQ(aacc.factor(), 2UL);
    ASSERT_GE(aacc.num_levels(), 2UL);

    // Level 0 gets all samples directly.
    ASSERT_EQ(aacc.count(0), dbl_5_n);
    check_near(aacc.mean(), dbl_5_mean, 1e-14);
    check_near(aacc.covariance(), dbl_5_var_mean, 1e-14);

    // Higher levels have fewer completed blocks.
    ASSERT_LE(aacc.count(1), aacc.count(0));
}

TEST_F(SimplemcAccs, CovarAccAutocorrFactory) {
    auto aacc = make_autocorr_covar_acc(dbl_5);
    ASSERT_EQ(aacc.count(0), dbl_5_n);
    check_near(aacc.mean(), dbl_5_mean, 1e-14);
}

// Test index-based accumulation for covar_acc.
TEST_F(SimplemcAccs, CovarAccIndexBased) {
    // accumulate only element 0 of 3-element accumulator
    covar_acc_static<double, 3> acc;
    for (const auto& v : vec_d_data) {
        acc[0] << v[0];
    }

    // only element 0 should have the correct mean, others are 0
    ASSERT_EQ(acc.count(), vec_d_n);
    check_near(acc.mean()[0], vec_d_mean[0], 1e-14);
    check_near(acc.mean()[1], 0.0, 1e-14);
    check_near(acc.mean()[2], 0.0, 1e-14);
}

TEST_F(SimplemcAccs, CovarAccAccumulateWithIndices) {
    // use accumulate(range, indices) to accumulate only element 0 and 2 of 3-element accumulator
    covar_acc_static<double, 3> acc_wel;
    covar_acc_static<double, 3, standard> acc_std;
    std::vector<long> idxs = { 0, 2 };
    for (const auto& v : vec_d_data) {
        Eigen::Vector2d sub(v[0], v[2]);
        acc_wel.accumulate(sub, idxs);
        acc_std.accumulate(sub, idxs);
    }

    // only elements 0 and 2 should have the correct mean, element 1 is 0
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    Eigen::Vector3d expected = { vec_d_mean[0], 0.0, vec_d_mean[2] };
    check_near(acc_wel.mean(), expected, 1e-14);
    check_near(acc_std.mean(), expected, 1e-14);
}

TEST_F(SimplemcAccs, CovarAccAccumulateConsecutive) {
    // use accumulate(range, start_index) to accumulate elements 1 and 2 of 3-element accumulator
    covar_acc_static<double, 3> acc_wel;
    covar_acc_static<double, 3, standard> acc_std;
    for (const auto& v : vec_d_data) {
        std::vector<double> sub = { v[1], v[2] };
        acc_wel.accumulate(sub, 1);
        acc_std.accumulate(sub, 1);
    }

    // only elements 1 and 2 should have the correct mean, element 0 is 0
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    Eigen::Vector3d expected = { 0.0, vec_d_mean[1], vec_d_mean[2] };
    check_near(acc_wel.mean(), expected, 1e-14);
    check_near(acc_std.mean(), expected, 1e-14);
}
