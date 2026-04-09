#include "./accs_fixture.hpp"

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/concepts.hpp>
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
TEST_F(SimplemcAccs, BatchAccConcepts) {
    // batch_acc with scalar types
    static_assert(covariance_accumulator<batch_acc<double>>);
    static_assert(covariance_accumulator<batch_acc<std::complex<double>>>);

    // batch_acc with static vector types
    static_assert(covariance_accumulator<batch_acc_static<double, 3>>);
    static_assert(covariance_accumulator<batch_acc_static<std::complex<double>, 3>>);

    // batch_acc with dynamic vector types
    static_assert(covariance_accumulator<batch_acc_dynamic<double>>);
    static_assert(covariance_accumulator<batch_acc_dynamic<std::complex<double>>>);

    // batch_acc with varalg::standard
    static_assert(covariance_accumulator<batch_acc<double, standard>>);
    static_assert(covariance_accumulator<batch_acc<cplx, standard>>);

    // multivalue_acc wrapping batch_acc
    static_assert(basic_accumulator<multivalue_acc<batch_acc<double>>>);
    static_assert(basic_accumulator<multivalue_acc<batch_acc<std::complex<double>>>>);
    static_assert(basic_accumulator<multivalue_acc<batch_acc_static<double, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<batch_acc_static<std::complex<double>, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<batch_acc_dynamic<double>>>);
    static_assert(basic_accumulator<multivalue_acc<batch_acc_dynamic<std::complex<double>>>>);
}

// Test empty accumulators.
TEST_F(SimplemcAccs, BatchAccEmpty) {
    batch_acc<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    ASSERT_TRUE(acc_sd.empty());
    ASSERT_EQ(acc_sd.count(), 0UL);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    batch_acc<cplx, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    ASSERT_TRUE(acc_sc.empty());
    ASSERT_EQ(acc_sc.count(), 0UL);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    batch_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    ASSERT_TRUE(acc_st_d.empty());
    ASSERT_EQ(acc_st_d.count(), 0UL);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    batch_acc_static<cplx, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    ASSERT_TRUE(acc_st_c.empty());
    ASSERT_EQ(acc_st_c.count(), 0UL);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    batch_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    ASSERT_TRUE(acc_dyn_d.empty());
    ASSERT_EQ(acc_dyn_d.count(), 0UL);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    batch_acc_dynamic<cplx, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    ASSERT_TRUE(acc_dyn_c.empty());
    ASSERT_EQ(acc_dyn_c.count(), 0UL);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Test scalar double accumulation.
TEST_F(SimplemcAccs, BatchAccScalarDouble) {
    batch_acc<double, standard> acc_std(1, 2);
    batch_acc<double, welford> acc_wel(1, 2);
    for (const auto& x : dbl_5) {
        acc_std << x;
        acc_wel << x;
    }

    ASSERT_EQ(acc_std.count(), dbl_5_n);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    ASSERT_FALSE(acc_std.empty());

    // Mean via mean_accumulator().
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);

    // mean_accumulator has correct count.
    auto macc = acc_std.mean_accumulator();
    ASSERT_EQ(macc.count(), dbl_5_n);
    check_near(macc.mean(), dbl_5_mean, 1e-14);
}

// Test scalar complex accumulation.
TEST_F(SimplemcAccs, BatchAccScalarComplex) {
    batch_acc<cplx, standard> acc_std(1, 2);
    batch_acc<cplx, welford> acc_wel(1, 2);
    for (const auto& z : cplx_5) {
        acc_std << z;
        acc_wel << z;
    }

    ASSERT_EQ(acc_std.count(), cplx_5_n);
    ASSERT_EQ(acc_wel.count(), cplx_5_n);
    check_near(acc_std.mean(), cplx_5_mean, 1e-14);
    check_near(acc_wel.mean(), cplx_5_mean, 1e-14);
}

// Test static vector double accumulation.
TEST_F(SimplemcAccs, BatchAccStaticVectorDouble) {
    batch_acc_static<double, 3, standard> acc_std(3, 2);
    batch_acc_static<double, 3, welford> acc_wel(3, 2);
    for (const auto& v : vec_d_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_d_n);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);
}

// Test static vector complex accumulation.
TEST_F(SimplemcAccs, BatchAccStaticVectorComplex) {
    batch_acc_static<cplx, 2, standard> acc_std(2, 2);
    batch_acc_static<cplx, 2, welford> acc_wel(2, 2);
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);
    check_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_near(acc_wel.mean(), vec_c_mean, 1e-14);
}

// Test dynamic vector double accumulation.
TEST_F(SimplemcAccs, BatchAccDynamicVectorDouble) {
    batch_acc_dynamic<double, standard> acc_std(3, 2);
    batch_acc_dynamic<double, welford> acc_wel(3, 2);
    for (const auto& v : vec_d_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_d_n);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.size(), 3);
    ASSERT_EQ(acc_wel.size(), 3);
    check_near(acc_std.mean(), vec_d_mean, 1e-14);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);
}

// Test dynamic vector complex accumulation.
TEST_F(SimplemcAccs, BatchAccDynamicVectorComplex) {
    batch_acc_dynamic<cplx, standard> acc_std(2, 2);
    batch_acc_dynamic<cplx, welford> acc_wel(2, 2);
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);
    ASSERT_EQ(acc_std.size(), 2);
    ASSERT_EQ(acc_wel.size(), 2);
    check_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_near(acc_wel.mean(), vec_c_mean, 1e-14);
}

// Test batch mechanics: count, num_full_batches, batch_count.
TEST_F(SimplemcAccs, BatchAccMechanics) {
    // Use m_b=2 (minimum) with known data.
    batch_acc<double> acc(1, 2);

    // Feed {1,2,3,4,5}.
    for (const auto& x : dbl_5) { acc << x; }

    // Total count is all samples.
    ASSERT_EQ(acc.count(), dbl_5_n);

    // num_full_batches is at least 1 (2 samples fill both
    // accumulating batches at bcount=1, producing 2 full
    // batches, then bcount doubles for subsequent fills).
    ASSERT_GE(acc.num_full_batches(), 1UL);

    // batch_count is a power of 2.
    auto bc = acc.batch_count();
    ASSERT_TRUE(bc > 0 && (bc & (bc - 1)) == 0);
}

// Test variance and covariance accessors.
TEST_F(SimplemcAccs, BatchAccVarianceAccessor) {
    // Accumulate enough to get full batches.
    batch_acc<double> acc(1, 2);
    for (const auto& x : dbl_5) { acc << x; }

    // var_accumulator builds from full batch means.
    if (acc.num_full_batches() >= 2) {
        auto vacc = acc.var_accumulator();
        ASSERT_GE(vacc.count(), 2UL);
    }
}

TEST_F(SimplemcAccs, BatchAccCovarianceAccessor) {
    batch_acc<double> acc(1, 2);
    for (const auto& x : dbl_5) { acc << x; }

    if (acc.num_full_batches() >= 2) {
        auto cacc = acc.covar_accumulator();
        ASSERT_GE(cacc.count(), 2UL);
    }
}

// Test resetting a batch accumulator.
TEST_F(SimplemcAccs, BatchAccReset) {
    batch_acc<double> acc(1, 2);
    for (const auto& x : dbl_5) { acc << x; }
    ASSERT_FALSE(acc.empty());

    acc.reset();
    ASSERT_TRUE(acc.empty());
    ASSERT_EQ(acc.count(), 0UL);
}

// Test factory function for batch_acc.
TEST_F(SimplemcAccs, BatchAccFactory) {
    // default (welford) varalg
    auto acc_wel = make_batch_acc(dbl_5, 2);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);

    // explicit standard varalg
    auto acc_std = make_batch_acc<standard>(dbl_5, 2);
    ASSERT_EQ(acc_std.count(), dbl_5_n);
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);

    // with a non-null shift t
    const double shift = 1.0;
    auto acc_shifted = make_batch_acc(dbl_5, 2, std::optional<double>(shift));
    ASSERT_EQ(acc_shifted.count(), dbl_5_n);
    check_near(acc_shifted.mean() + shift, dbl_5_mean, 1e-14);
}

// Test factory function for batch_acc with vectors.
TEST_F(SimplemcAccs, BatchAccFactoryVector) {
    // default (welford) varalg
    auto acc_wel = make_batch_acc(vec_d_data, 2);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    check_near(acc_wel.mean(), vec_d_mean, 1e-14);

    // explicit standard varalg
    auto acc_std = make_batch_acc<standard>(vec_d_data, 2);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_near(acc_std.mean(), vec_d_mean, 1e-14);

    // with a non-null vector shift t
    const Eigen::Vector3d shift(1.0, 1.0, 1.0);
    auto acc_shifted = make_batch_acc(vec_d_data, 2, std::optional<Eigen::Vector3d>(shift));
    ASSERT_EQ(acc_shifted.count(), vec_d_n);
    const Eigen::Vector3d shifted_mean = (acc_shifted.mean() + shift).eval();
    check_near(shifted_mean, vec_d_mean, 1e-14);
}

// Test that make_batch_acc throws on an empty range.
TEST_F(SimplemcAccs, BatchAccFactoryEmptyRange) {
    std::vector<double> empty;
    EXPECT_THROW((void)make_batch_acc(empty, 2), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_vec;
    EXPECT_THROW((void)make_batch_acc(empty_vec, 2), simplemc_exception);
}

// Test that constructing dynamic batch_acc with invalid size throws.
TEST_F(SimplemcAccs, BatchAccDynamicConstructorInvalid) {
    EXPECT_THROW(batch_acc_dynamic<double>(0, 2), simplemc_exception);
}

// Test index-based accumulation for batch_acc.
TEST_F(SimplemcAccs, BatchAccIndexBased) {
    batch_acc_static<double, 3> acc(3, 2);
    for (const auto& v : vec_d_data) {
        acc[0] << v[0];
    }
    ASSERT_EQ(acc.count(), vec_d_n);
}

// Test accumulate(range, indices) for batch_acc.
TEST_F(SimplemcAccs, BatchAccAccumulateWithIndices) {
    // accumulate only elements 0 and 2 of 3-element batch accumulator
    batch_acc_static<double, 3> acc_wel(3, 2);
    batch_acc_static<double, 3, standard> acc_std(3, 2);
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

// Test accumulate(range, start_index) for batch_acc.
TEST_F(SimplemcAccs, BatchAccAccumulateConsecutive) {
    // accumulate elements 1 and 2 of 3-element batch accumulator
    batch_acc_static<double, 3> acc_wel(3, 2);
    batch_acc_static<double, 3, standard> acc_std(3, 2);
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

// Test multivalue_acc wrapper for batch_acc.
TEST_F(SimplemcAccs, BatchAccMultivalue) {
    batch_acc_static<double, 3> acc(3, 2);
    auto mva = acc.make_mva();

    for (const auto& v : vec_d_data) {
        for (int i = 0; i < 3; ++i) {
            mva[i] << v[i];
        }
        mva.increment_count();
        acc.check_and_advance();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    check_near(acc.mean(), vec_d_mean, 1e-14);
}
