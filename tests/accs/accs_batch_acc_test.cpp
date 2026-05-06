#include "./accs_test_traits.hpp"

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/concepts.hpp>
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
class SimplemcAccsBatch : public ::testing::Test {};
TYPED_TEST_SUITE(SimplemcAccsBatch, AllAccTypes);

// Test that accumulator concepts are satisfied.
TYPED_TEST(SimplemcAccsBatch, Concepts) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    static_assert(batch_accumulator<batch_acc<T, A>>);
    static_assert(basic_accumulator<multivalue_acc<batch_acc<T, A>>>);
}

// Test empty accumulator state.
TYPED_TEST(SimplemcAccsBatch, Empty) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    auto acc = make_empty_batch<acc_t, T>(4);
    ASSERT_EQ(acc.size(), vec_size_v<T>);
    check_acc_empty(acc);
}

// Test accumulation of N=100 deterministic samples for different m_b values.
TYPED_TEST(SimplemcAccsBatch, Accumulation) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto data = make_data<T>();
    for (auto m_b : { 2UL, 4UL, 8UL, 16UL }) {
        auto acc = make_empty_batch<batch_acc<T, A>, T>(m_b);
        check_acc_empty(acc);
        for (const auto& x : data) {
            acc << x;
        }
        check_batch_acc(acc, m_b);
    }
}

// Test reset.
TYPED_TEST(SimplemcAccsBatch, Reset) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty_batch<acc_t, T>(2);
    for (const auto& x : data) {
        acc << x;
    }
    ASSERT_FALSE(acc.empty());

    acc.reset();
    check_acc_empty(acc);
}

// Test data constructor.
TYPED_TEST(SimplemcAccsBatch, DataConstructor) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    auto data = make_data<T>();
    for (auto m_b : { 2ul, 4ul, 6ul, 8ul, 10ul }) {
        auto acc = make_empty_batch<acc_t, T>(m_b);
        for (const auto& x : data) {
            acc << x;
        }

        acc_t acc_copy(acc.batch_vector_full(), acc.batch_vector_accumulating());
        check_acc_equal(acc_copy, acc);
    }
}

// Test factory function.
TYPED_TEST(SimplemcAccsBatch, Factory) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;

    auto acc = make_batch_acc<A>(make_data<T>(), 2);
    check_batch_acc(acc, 2);
}

// Test factory function with shift: factory should produce the same result
// as manually accumulating (sample - shift).
TYPED_TEST(SimplemcAccsBatch, FactoryWithShift) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    auto data = make_data<T>();
    T shift = make_shift<T>();
    auto acc = make_batch_acc<A>(data, 2, std::optional<T>(shift));

    auto acc_ref = make_empty_batch<acc_t, T>(2);
    for (const auto& x : data) {
        acc_ref << (x - shift);
    }
    check_acc_equal(acc, acc_ref);
}

// Test index-based accumulation (vector types only).
// Reference: accumulate full vectors with zeros in non-indexed positions.
TYPED_TEST(SimplemcAccsBatch, IndexBased) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty_batch<acc_t, T>(2);
        auto acc_ref = make_empty_batch<acc_t, T>(2);
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
TYPED_TEST(SimplemcAccsBatch, AccumulateWithIndices) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty_batch<acc_t, T>(2);
        auto acc_ref = make_empty_batch<acc_t, T>(2);
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
TYPED_TEST(SimplemcAccsBatch, AccumulateConsecutive) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();
        auto acc = make_empty_batch<acc_t, T>(2);
        auto acc_ref = make_empty_batch<acc_t, T>(2);
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

// Test multivalue_acc wrapper for batch_acc.
TYPED_TEST(SimplemcAccsBatch, Multivalue) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;

    if constexpr (!is_scalar_sample_v<T>) {
        auto data = make_data<T>();

        for (auto m_b : { 2ul, 4ul, 6ul, 8ul, 10ul }) {
            auto acc_ref = make_empty_batch<acc_t, T>(m_b);
            auto acc = make_empty_batch<acc_t, T>(m_b);

            for (const auto& v : data) {
                acc_ref << v;
                auto mva = acc.make_mva();
                for (long i = 0; i < vec_size_v<T>; ++i) {
                    mva[i] << component(v, i);
                }
                mva.increment_count();
                acc.check_and_advance();
            }

            check_acc_equal(acc, acc_ref);
        }
    }
}

// Non-typed tests for edge cases.
TEST(BatchAccBasic, DynamicConstructorInvalid) {
    EXPECT_THROW(batch_acc_dynamic<double>(0, 2), simplemc_exception);
    EXPECT_THROW(batch_acc_dynamic<std::complex<double>>(0, 2), simplemc_exception);
}

TEST(BatchAccBasic, FactoryEmptyRange) {
    std::vector<double> empty_d;
    EXPECT_THROW((void)make_batch_acc(empty_d, 2), simplemc_exception);
    std::vector<Eigen::Vector3d> empty_v;
    EXPECT_THROW((void)make_batch_acc(empty_v, 2), simplemc_exception);
}
