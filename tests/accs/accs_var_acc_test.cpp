#include "./accs_test_traits.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <optional>
#include <vector>

namespace {

using namespace simplemc;
using namespace test_detail;

} // namespace

// Typed test suite over all 12 accumulator type combinations.
template <typename Tag>
class SimplemcAccsVar : public ::testing::Test {};
TYPED_TEST_SUITE(SimplemcAccsVar, AllAccTypes);

// Test that accumulator concepts are satisfied.
TYPED_TEST(SimplemcAccsVar, Concepts) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    static_assert(variance_accumulator<var_acc<T, A>>);
    static_assert(basic_accumulator<multivalue_acc<var_acc<T, A>>>);
    static_assert(variance_accumulator<block_acc<var_acc<T, A>>>);
    static_assert(variance_accumulator<autocorr_acc<var_acc<T, A>>>);
}

// Test empty accumulator state.
TYPED_TEST(SimplemcAccsVar, Empty) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto acc = make_empty<acc_t, T>();
    ASSERT_EQ(acc.size(), vec_size_v<T>);
    check_acc_empty(acc);

    // merging two empty accumulators stays empty
    auto acc2 = make_empty<acc_t, T>();
    acc << acc2;
    check_acc_empty(acc);
}

// Test accumulation of N=100 deterministic samples.
TYPED_TEST(SimplemcAccsVar, Accumulation) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (const auto& x : data) {
        acc << x;
    }

    check_acc_results<T>(acc);
}

// Test merging two accumulators (split 60/40).
TYPED_TEST(SimplemcAccsVar, Merge) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    auto acc1 = make_empty<acc_t, T>();
    auto acc2 = make_empty<acc_t, T>();
    auto acc_exp = make_empty<acc_t, T>();

    for (int i = 0; i < 60; ++i) {
        acc1 << data[i];
        acc_exp << data[i];
    }
    for (int i = 60; i < N; ++i) {
        acc2 << data[i];
        acc_exp << data[i];
    }
    ASSERT_EQ(acc1.count(), 60);
    ASSERT_EQ(acc2.count(), 40);
    ASSERT_EQ(acc_exp.count(), N);

    acc1 << acc2;
    ASSERT_EQ(acc2.count(), 40); // source unchanged
    check_acc_equal(acc1, acc_exp);
}

// Test reset.
TYPED_TEST(SimplemcAccsVar, Reset) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (const auto& x : data) {
        acc << x;
    }
    ASSERT_FALSE(acc.empty());

    acc.reset();
    check_acc_empty(acc);
}

// Test data constructor.
TYPED_TEST(SimplemcAccsVar, DataConstructor) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (const auto& x : data) {
        acc << x;
    }

    if constexpr (is_complex_sample_v<T>) {
        acc_t acc_copy(acc.mdata(), acc.rdata(), acc.idata(), acc.cdata(), acc.count());
        check_acc_equal(acc_copy, acc);
    } else {
        acc_t acc_copy(acc.mdata(), acc.cdata(), acc.count());
        check_acc_equal(acc_copy, acc);
    }
}

// Test factory function.
TYPED_TEST(SimplemcAccsVar, Factory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto acc = make_var_acc<A>(make_data<T>());
    check_acc_results<T>(acc);
}

// Test factory function with shift: factory should produce the same result
// as manually accumulating (sample - shift).
TYPED_TEST(SimplemcAccsVar, FactoryWithShift) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    T shift = make_shift<T>();
    auto acc = make_var_acc<A>(data, std::optional<T>(shift));

    auto acc_ref = make_empty<acc_t, T>();
    for (const auto& x : data) {
        acc_ref << (x - shift);
    }
    check_acc_equal(acc, acc_ref);
}

// Test index-based accumulation (vector types only).
// Reference: accumulate full vectors with zeros in non-indexed positions.
TYPED_TEST(SimplemcAccsVar, IndexBased) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty<acc_t, T>();
        auto acc_ref = make_empty<acc_t, T>();
        T ref = make_zero_sample<T>();
        for (const auto& v : data) {
            acc[1] << v[1];
            ref[1] = v[1];
            acc_ref << ref;
        }
        check_acc_equal(acc, acc_ref);
    }
}

// Test accumulate with arbitrary indices (3-element vectors).
// Reference: accumulate full vectors with zeros in non-indexed positions.
TYPED_TEST(SimplemcAccsVar, AccumulateWithIndices) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty<acc_t, T>();
        auto acc_ref = make_empty<acc_t, T>();
        std::vector<long> idxs = { 0, 2 };
        T ref = make_zero_sample<T>();
        for (const auto& v : data) {
            auto sub = std::vector<typename T::Scalar> { v[0], v[2] };
            acc.accumulate(sub, idxs);
            ref[0] = v[0];
            ref[2] = v[2];
            acc_ref << ref;
        }
        check_acc_equal(acc, acc_ref);
    }
}

// Test accumulate with consecutive indices (3-element vectors).
// Reference: accumulate full vectors with zeros in non-indexed positions.
TYPED_TEST(SimplemcAccsVar, AccumulateConsecutive) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty<acc_t, T>();
        auto acc_ref = make_empty<acc_t, T>();
        T ref = make_zero_sample<T>();
        for (const auto& v : data) {
            auto sub = std::vector<typename T::Scalar> { v[1], v[2] };
            acc.accumulate(sub, 1);
            ref[1] = v[1];
            ref[2] = v[2];
            acc_ref << ref;
        }
        check_acc_equal(acc, acc_ref);
    }
}

// Test multivalue_acc wrapper: element-by-element accumulation.
TYPED_TEST(SimplemcAccsVar, Multivalue) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty<acc_t, T>();

        for (const auto& v : data) {
            auto mva = acc.make_mva();
            for (long i = 0; i < vec_size_v<T>; ++i) {
                mva[i] << component(v, i);
            }
            mva.increment_count();
        }

        check_acc_results<T>(acc);
    }
}

// Test multivalue_acc vector stream operator.
TYPED_TEST(SimplemcAccsVar, MultivalueVectorStream) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty<acc_t, T>();

        for (const auto& v : data) {
            auto mva = acc.make_mva();
            mva << v;
            mva.increment_count();
        }

        check_acc_results<T>(acc);
    }
}

// Test multivalue_acc accessors.
TYPED_TEST(SimplemcAccsVar, MultivalueAccessors) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto acc = make_empty<acc_t, T>();
    auto mva = acc.make_mva();

    ASSERT_EQ(mva.size(), vec_size_v<T>);
    ASSERT_EQ(mva.count(), 0);
    ASSERT_TRUE(mva.empty());
    ASSERT_EQ(&mva.accumulator(), &acc);

    const auto& mva_const = mva;
    ASSERT_EQ(&mva_const.accumulator(), &acc);

    mva[0] << component(make_data<T>()[0], 0);
    mva.increment_count();
    ASSERT_EQ(mva.count(), 1);
    ASSERT_FALSE(mva.empty());

    mva[0] << component(make_data<T>()[1], 0);
    mva[0] << component(make_data<T>()[2], 0);
    mva.increment_count(2);
    ASSERT_EQ(mva.count(), 3);
}

// Test block_acc wrapping var_acc for different block sizes.
TYPED_TEST(SimplemcAccsVar, Block) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto bl_size : { 1ul, 2ul, 7ul, 10ul }) {
        auto bacc = make_empty_block<block_acc<var_acc<T, A>>, T>(bl_size);
        check_acc_empty(bacc);
        for (const auto& x : data) {
            bacc << x;
        }
        check_block_acc(bacc, bl_size);
    }
}

// Test block_acc factory for different block sizes.
TYPED_TEST(SimplemcAccsVar, BlockFactory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto bl_size : { 1ul, 2ul, 7ul, 10ul }) {
        auto bacc = make_block_var_acc<A>(data, bl_size);
        check_block_acc(bacc, bl_size);
    }
}

// Test autocorr_acc wrapping var_acc for different factors.
TYPED_TEST(SimplemcAccsVar, Autocorr) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto c : { 2, 3, 4 }) {
        auto aacc = make_empty_autocorr<autocorr_acc<var_acc<T, A>>, T>(c);
        check_acc_empty(aacc);
        for (const auto& x : data) {
            aacc << x;
        }
        check_autocorr_acc(aacc, c);
    }
}

// Test autocorr_acc factory.
TYPED_TEST(SimplemcAccsVar, AutocorrFactory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    auto aacc = make_autocorr_var_acc<A>(data);

    check_autocorr_acc(aacc, 2);
}

// Non-typed tests for edge cases.
TEST(SimplemcAccsVarBasic, DynamicConstructorInvalid) {
    EXPECT_THROW(var_acc_dynamic<double>(0), simplemc_exception);
    EXPECT_THROW(var_acc_dynamic<std::complex<double>>(0), simplemc_exception);

    Eigen::VectorXd empty_vec(0);
    EXPECT_THROW((var_acc_dynamic<double>(empty_vec, empty_vec, 0)), simplemc_exception);
}

TEST(SimplemcAccsVarBasic, FactoryEmptyRange) {
    std::vector<double> empty_d;
    EXPECT_THROW((void)make_var_acc(empty_d), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_v;
    EXPECT_THROW((void)make_var_acc(empty_v), simplemc_exception);
}
