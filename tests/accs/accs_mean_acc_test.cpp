#include "./accs_fixture.hpp"

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/mean_acc.hpp>
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
TEST_F(SimplemcAccs, MeanAccConcepts) {
    // mean_acc with scalar types
    static_assert(mean_accumulator<mean_acc<double>>);
    static_assert(mean_accumulator<mean_acc<std::complex<double>>>);

    // mean_acc with static vector types
    static_assert(mean_accumulator<mean_acc_static<double, 3>>);
    static_assert(mean_accumulator<mean_acc_static<std::complex<double>, 3>>);

    // mean_acc with dynamic vector types
    static_assert(mean_accumulator<mean_acc_dynamic<double>>);
    static_assert(mean_accumulator<mean_acc_dynamic<std::complex<double>>>);

    // multivalue_acc wrapping mean_acc
    static_assert(basic_accumulator<multivalue_acc<mean_acc<double>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc<std::complex<double>>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc_static<double, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc_static<std::complex<double>, 3>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc_dynamic<double>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc_dynamic<std::complex<double>>>>);
}

// Test empty accumulators.
TEST_F(SimplemcAccs, MeanAccEmpty) {
    mean_acc<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_empty(acc_sd);
    acc_sd << acc_sd;
    check_empty(acc_sd);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    mean_acc<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_empty(acc_sc);
    acc_sc << acc_sc;
    check_empty(acc_sc);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    mean_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_empty(acc_st_d);
    acc_st_d << acc_st_d;
    check_empty(acc_st_d);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    mean_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_empty(acc_st_c);
    acc_st_c << acc_st_c;
    check_empty(acc_st_c);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    mean_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_empty(acc_dyn_d);
    acc_dyn_d << acc_dyn_d;
    check_empty(acc_dyn_d);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    mean_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_empty(acc_dyn_c);
    acc_dyn_c << acc_dyn_c;
    check_empty(acc_dyn_c);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Test scalar double accumulation.
TEST_F(SimplemcAccs, MeanAccScalarDouble) {
    mean_acc<double, standard> acc_std;
    mean_acc<double, welford> acc_wel;
    for (const auto& x : dbl_5) {
        acc_std << x;
        acc_wel << x;
    }

    ASSERT_EQ(acc_std.count(), dbl_5_n);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    ASSERT_FALSE(acc_std.empty());
    ASSERT_FALSE(acc_wel.empty());
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);
}

// Test scalar complex accumulation.
TEST_F(SimplemcAccs, MeanAccScalarComplex) {
    mean_acc<cplx, standard> acc_std;
    mean_acc<cplx, welford> acc_wel;
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
TEST_F(SimplemcAccs, MeanAccStaticVectorDouble) {
    mean_acc_static<double, 3, welford> acc_wel;
    mean_acc_static<double, 3, standard> acc_std;
    for (const auto& v : vec_d_data) {
        acc_wel << v;
        acc_std << v;
    }

    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_range_near(acc_wel.mean(), vec_d_mean, 1e-14);
    check_range_near(acc_std.mean(), vec_d_mean, 1e-14);
}

// Test static vector complex accumulation.
TEST_F(SimplemcAccs, MeanAccStaticVectorComplex) {
    mean_acc_static<cplx, 2, standard> acc_std;
    mean_acc_static<cplx, 2, welford> acc_wel;
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);
    check_range_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_range_near(acc_wel.mean(), vec_c_mean, 1e-14);
}

// Test dynamic vector double accumulation.
TEST_F(SimplemcAccs, MeanAccDynamicVectorDouble) {
    mean_acc_dynamic<double, welford> acc_wel(3);
    mean_acc_dynamic<double, standard> acc_std(3);
    for (const auto& v : vec_d_data) {
        acc_wel << v;
        acc_std << v;
    }

    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_range_near(acc_wel.mean(), vec_d_mean, 1e-14);
    check_range_near(acc_std.mean(), vec_d_mean, 1e-14);
}

// Test dynamic vector complex accumulation.
TEST_F(SimplemcAccs, MeanAccDynamicVectorComplex) {
    mean_acc_dynamic<cplx, standard> acc_std(2);
    mean_acc_dynamic<cplx, welford> acc_wel(2);
    for (const auto& v : vec_c_data) {
        acc_std << v;
        acc_wel << v;
    }

    ASSERT_EQ(acc_std.count(), vec_c_n);
    ASSERT_EQ(acc_wel.count(), vec_c_n);
    check_range_near(acc_std.mean(), vec_c_mean, 1e-14);
    check_range_near(acc_wel.mean(), vec_c_mean, 1e-14);
}

// Test merging two mean_acc accumulators.
TEST_F(SimplemcAccs, MeanAccMergeScalar) {
    // split dbl_5 into two halves, merge, verify
    mean_acc<double, standard> acc1_std, acc2_std;
    mean_acc<double, welford> acc1_wel, acc2_wel;
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
}

TEST_F(SimplemcAccs, MeanAccMergeVector) {
    // split vec_d_data into two halves, merge, verify
    mean_acc_static<double, 3, standard> acc1_std, acc2_std;
    mean_acc_dynamic<double, welford> acc1_wel(3), acc2_wel(3);
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
    check_range_near(acc1_std.mean(), vec_d_mean, 1e-14);
    check_range_near(acc1_wel.mean(), vec_d_mean, 1e-14);
}

// Test reseting an accumulator.
TEST_F(SimplemcAccs, MeanAccReset) {
    mean_acc<double> acc;
    for (const auto& x : dbl_5) {
        acc << x;
    }
    ASSERT_FALSE(acc.empty());

    acc.reset();
    check_empty(acc);
}

// Test constructing a mean_acc from mdata and count.
TEST_F(SimplemcAccs, MeanAccDataConstructor) {
    // scalar data constructor (valid case)
    mean_acc<double> acc;
    for (const auto& x : dbl_5) {
        acc << x;
    }
    mean_acc<double> acc_copy(acc.mdata(), acc.count());
    ASSERT_EQ(acc_copy.count(), acc.count());
    ASSERT_EQ(acc_copy.mdata(), acc.mdata());
    check_near(acc_copy.mean(), dbl_5_mean, 1e-14);

    // dynamic data constructor (valid case)
    mean_acc_dynamic<double> acc_dyn(3);
    for (const auto& v : vec_d_data) {
        acc_dyn << v;
    }
    mean_acc_dynamic<double> acc_dyn_copy(acc_dyn.mdata(), acc_dyn.count());
    ASSERT_EQ(acc_dyn_copy.count(), acc_dyn.count());
    ASSERT_TRUE(acc_dyn_copy.mdata() == acc_dyn.mdata());
    check_range_near(acc_dyn_copy.mean(), vec_d_mean, 1e-14);
}

// Test that constructing dynamic mean_acc with invalid sizes throws.
// Note: only m == 0 is tested here; m < 0 would trigger an Eigen internal assertion
// (in the member initializer) before our check runs.
TEST_F(SimplemcAccs, MeanAccDynamicConstructorInvalid) {
    EXPECT_THROW(mean_acc_dynamic<double>(0), simplemc_exception);

    // data constructor: zero-size vec_type triggers the same exception
    Eigen::VectorXd empty_vec(0);
    EXPECT_THROW((mean_acc_dynamic<double>(empty_vec, 0)), simplemc_exception);
}

// Test index-based accumulation for mean_acc.
TEST_F(SimplemcAccs, MeanAccIndexBased) {
    // accumulate only element 1 of 3-element accumulator
    mean_acc_static<double, 3> acc;
    for (const auto& v : vec_d_data) {
        acc[1] << v[1];
    }

    // only element 1 should have the correct mean, others are 0
    ASSERT_EQ(acc.count(), vec_d_n);
    Eigen::Vector3d expected = { 0.0, vec_d_mean[1], 0.0 };
    check_range_near(acc.mean(), expected, 1e-14);
}

TEST_F(SimplemcAccs, MeanAccAccumulateWithIndices) {
    // use accumulate(range, indices) to accumulate only element 0 and 2 of 3-element accumulator
    mean_acc_static<double, 3> acc_wel;
    mean_acc_static<double, 3, standard> acc_std;
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
    check_range_near(acc_wel.mean(), expected, 1e-14);
    check_range_near(acc_std.mean(), expected, 1e-14);
}

TEST_F(SimplemcAccs, MeanAccAccumulateConsecutive) {
    // use accumulate(range, start_index) to accumulate elements 1 and 2 of 3-element accumulator
    mean_acc_static<double, 3> acc_wel;
    mean_acc_static<double, 3, standard> acc_std;
    for (const auto& v : vec_d_data) {
        std::vector<double> sub = { v[1], v[2] };
        acc_wel.accumulate(sub, 1);
        acc_std.accumulate(sub, 1);
    }

    // only elements 1 and 2 should have the correct mean, element 0 is 0
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    Eigen::Vector3d expected = { 0.0, vec_d_mean[1], vec_d_mean[2] };
    check_range_near(acc_wel.mean(), expected, 1e-14);
    check_range_near(acc_std.mean(), expected, 1e-14);
}

// Test factory function for mean_acc.
TEST_F(SimplemcAccs, MeanAccFactoryScalar) {
    // default (welford) varalg
    auto acc_wel = make_mean_acc(dbl_5);
    ASSERT_EQ(acc_wel.count(), dbl_5_n);
    check_near(acc_wel.mean(), dbl_5_mean, 1e-14);

    // explicit standard varalg
    auto acc_std = make_mean_acc<standard>(dbl_5);
    ASSERT_EQ(acc_std.count(), dbl_5_n);
    check_near(acc_std.mean(), dbl_5_mean, 1e-14);

    // with a non-null shift t
    const double shift = 1.0;
    auto acc_shifted = make_mean_acc(dbl_5, std::optional<double>(shift));
    ASSERT_EQ(acc_shifted.count(), dbl_5_n);
    check_near(acc_shifted.mean() + shift, dbl_5_mean, 1e-14);
}

TEST_F(SimplemcAccs, MeanAccFactoryVector) {
    // default (welford) varalg
    auto acc_wel = make_mean_acc(vec_d_data);
    ASSERT_EQ(acc_wel.count(), vec_d_n);
    check_range_near(acc_wel.mean(), vec_d_mean, 1e-14);

    // explicit standard varalg
    auto acc_std = make_mean_acc<standard>(vec_d_data);
    ASSERT_EQ(acc_std.count(), vec_d_n);
    check_range_near(acc_std.mean(), vec_d_mean, 1e-14);

    // with a non-null vector shift t
    const Eigen::Vector3d shift(1.0, 1.0, 1.0);
    auto acc_shifted = make_mean_acc(vec_d_data, std::optional<Eigen::Vector3d>(shift));
    ASSERT_EQ(acc_shifted.count(), vec_d_n);
    const Eigen::Vector3d shifted_mean = (acc_shifted.mean() + shift).eval();
    check_range_near(shifted_mean, vec_d_mean, 1e-14);
}

// Test that make_mean_acc throws on an empty range (detail::make_acc exception path).
TEST_F(SimplemcAccs, MeanAccFactoryEmptyRange) {
    std::vector<double> empty;
    EXPECT_THROW((void)make_mean_acc(empty), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_vec;
    EXPECT_THROW((void)make_mean_acc(empty_vec), simplemc_exception);
}

// Test multivalue_acc wrapper for mean_acc.
TEST_F(SimplemcAccs, MeanAccMultivalue) {
    mean_acc_static<double, 3> acc_wel;
    mean_acc_static<double, 3, standard> acc_std;
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
    check_range_near(acc_wel.mean(), vec_d_mean, 1e-14);
    check_range_near(acc_std.mean(), vec_d_mean, 1e-14);
}

// Test multivalue_acc vector stream operator (operator<<(W v)).
TEST_F(SimplemcAccs, MeanAccMultivalueVectorStream) {
    mean_acc_static<double, 3> acc;
    auto mva = acc.make_mva();

    for (const auto& v : vec_d_data) {
        mva << v;
        mva.increment_count();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    check_range_near(acc.mean(), vec_d_mean, 1e-14);
}

// Test multivalue_acc accumulate with consecutive indices.
TEST_F(SimplemcAccs, MeanAccMultivalueAccumulateConsecutive) {
    mean_acc_static<double, 3> acc;
    auto mva = acc.make_mva();

    for (const auto& v : vec_d_data) {
        std::vector<double> sub = { v[1], v[2] };
        mva.accumulate(sub, 1);
        mva.increment_count();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    Eigen::Vector3d expected = { 0.0, vec_d_mean[1], vec_d_mean[2] };
    check_range_near(acc.mean(), expected, 1e-14);
}

// Test multivalue_acc accumulate with arbitrary indices.
TEST_F(SimplemcAccs, MeanAccMultivalueAccumulateWithIndices) {
    mean_acc_static<double, 3> acc;
    auto mva = acc.make_mva();
    std::vector<long> idxs = { 0, 2 };

    for (const auto& v : vec_d_data) {
        Eigen::Vector2d sub(v[0], v[2]);
        mva.accumulate(sub, idxs);
        mva.increment_count();
    }

    ASSERT_EQ(acc.count(), vec_d_n);
    Eigen::Vector3d expected = { vec_d_mean[0], 0.0, vec_d_mean[2] };
    check_range_near(acc.mean(), expected, 1e-14);
}

// Test multivalue_acc accessors: size(), count(), empty(), accumulator().
TEST_F(SimplemcAccs, MeanAccMultivalueAccessors) {
    mean_acc_static<double, 3> acc;
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
