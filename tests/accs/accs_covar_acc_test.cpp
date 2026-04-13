#include "./accs_test_traits.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
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
class SimplemcAccsCovar : public ::testing::Test {};
TYPED_TEST_SUITE(SimplemcAccsCovar, AllAccTypes);

// Test that accumulator concepts are satisfied.
TYPED_TEST(SimplemcAccsCovar, Concepts) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    static_assert(covariance_accumulator<covar_acc<T, A>>);
    static_assert(basic_accumulator<multivalue_acc<covar_acc<T, A>>>);
    static_assert(covariance_accumulator<block_acc<covar_acc<T, A>>>);
    static_assert(covariance_accumulator<autocorr_acc<covar_acc<T, A>>>);
}

// Test empty accumulator state.
TYPED_TEST(SimplemcAccsCovar, Empty) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

    auto acc = make_empty<acc_t, T>();
    ASSERT_EQ(acc.size(), vec_size_v<T>);
    check_acc_empty(acc);

    // merging two empty accumulators stays empty
    auto acc2 = make_empty<acc_t, T>();
    acc << acc2;
    check_acc_empty(acc);
}

// Test accumulation of N=100 deterministic samples.
TYPED_TEST(SimplemcAccsCovar, Accumulation) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (const auto& x : data) {
        acc << x;
    }

    check_acc_results<T>(acc);
}

// Test merging two accumulators (split 60/40).
TYPED_TEST(SimplemcAccsCovar, Merge) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

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
TYPED_TEST(SimplemcAccsCovar, Reset) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

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
TYPED_TEST(SimplemcAccsCovar, DataConstructor) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

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
TYPED_TEST(SimplemcAccsCovar, Factory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto acc = make_covar_acc<A>(make_data<T>());
    check_acc_results<T>(acc);
}

// Test factory function with shift: factory should produce the same result
// as manually accumulating (sample - shift).
TYPED_TEST(SimplemcAccsCovar, FactoryWithShift) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

    auto data = make_data<T>();
    T shift = make_shift<T>();
    auto acc = make_covar_acc<A>(data, std::optional<T>(shift));

    auto acc_ref = make_empty<acc_t, T>();
    for (const auto& x : data) {
        acc_ref << (x - shift);
    }
    check_acc_equal(acc, acc_ref);
}

// Test index-based accumulation (vector types only).
// Reference: accumulate full vectors with zeros in non-indexed positions.
TYPED_TEST(SimplemcAccsCovar, IndexBased) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

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
TYPED_TEST(SimplemcAccsCovar, AccumulateWithIndices) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

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
TYPED_TEST(SimplemcAccsCovar, AccumulateConsecutive) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

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

// Test block_acc wrapping var_acc for different block sizes.
TYPED_TEST(SimplemcAccsCovar, Block) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto bl_size : { 1ul, 2ul, 7ul, 10ul }) {
        auto bacc = make_empty_block<block_acc<covar_acc<T, A>>, T>(bl_size);
        check_acc_empty(bacc);
        for (const auto& x : data) {
            bacc << x;
        }
        check_block_acc(bacc, bl_size);
    }
}

// Test block_acc factory for different block sizes.
TYPED_TEST(SimplemcAccsCovar, BlockFactory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto bl_size : { 1ul, 2ul, 7ul, 10ul }) {
        auto bacc = make_block_covar_acc<A>(data, bl_size);
        check_block_acc(bacc, bl_size);
    }
}

// Test autocorr_acc wrapping covar_acc for different factors.
TYPED_TEST(SimplemcAccsCovar, Autocorr) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto c : { 2, 3, 4 }) {
        auto aacc = make_empty_autocorr<autocorr_acc<covar_acc<T, A>>, T>(c);
        check_acc_empty(aacc);
        for (const auto& x : data) {
            aacc << x;
        }
        check_autocorr_acc(aacc, c);
    }
}

// Test autocorr_acc factory.
TYPED_TEST(SimplemcAccsCovar, AutocorrFactory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    auto aacc = make_autocorr_covar_acc<A>(data);

    check_autocorr_acc(aacc, 2);
}

// Non-typed tests for edge cases.
TEST(CovarAccBasic, DynamicConstructorInvalid) {
    EXPECT_THROW(covar_acc_dynamic<double>(0), simplemc_exception);
    EXPECT_THROW(covar_acc_dynamic<std::complex<double>>(0), simplemc_exception);

    Eigen::VectorXd empty_vec(0);
    Eigen::MatrixXd empty_mat(0, 0);
    EXPECT_THROW((covar_acc_dynamic<double>(empty_vec, empty_mat, 0)), simplemc_exception);
}

TEST(CovarAccBasic, FactoryEmptyRange) {
    std::vector<double> empty_d;
    EXPECT_THROW((void)make_covar_acc(empty_d), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_v;
    EXPECT_THROW((void)make_covar_acc(empty_v), simplemc_exception);
}
