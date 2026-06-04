#include <simplemc/mc.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <vector>

using namespace simplemc;

namespace {

// A toy measurement that increments a shared counter so we can observe `measure()` calls
// across copies and through type erasure.
struct counter_meas {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

// A second, distinct conforming type used for heterogeneous-storage tests.
struct doubling_meas {
    std::shared_ptr<int> count = std::make_shared<int>(1);
    void measure() { *count *= 2; }
};

// Negative case for the concept: missing `measure()`.
struct not_a_meas {
    int x = 0;
};

// A `mc_measurement` that is not copy-constructible: the wrapper must reject it at construction
// even though the concept itself accepts it.
struct move_only_meas {
    std::unique_ptr<int> count = std::make_unique<int>(0);
    void measure() { ++*count; }
};

} // namespace

// Check mc_measurement concept.
static_assert(mc_measurement<counter_meas>);
static_assert(mc_measurement<doubling_meas>);
static_assert(!mc_measurement<not_a_meas>);
static_assert(!mc_measurement<int>);

// the wrapper itself satisfies the concept — it forwards `measure()`
static_assert(mc_measurement<measurement>);

// the concept is a pure role description: a move-only measurement still satisfies it
// (but the wrapper rejects it because of the copyability requirement)
static_assert(mc_measurement<move_only_meas>);
static_assert(!std::is_constructible_v<measurement, int>);
static_assert(!std::is_constructible_v<measurement, not_a_meas>);

TEST(MCMeasurement, WrapsAndForwardsMeasure) {
    counter_meas src;
    auto count = src.count;

    measurement m { src };
    m.measure();

    EXPECT_EQ(*count, 1);
}

TEST(MCMeasurement, CopyProducesIndependentValueSharingCapturedState) {
    counter_meas src;
    auto count = src.count;

    measurement a { src };
    measurement b { a }; // deep copy via clone()

    a.measure();
    b.measure();

    // both wrappers' models hold their own copy of `counter_meas`
    EXPECT_EQ(*count, 2);
}

TEST(MCMeasurement, MoveTransfersOwnership) {
    counter_meas src;
    auto count = src.count;

    measurement a { src };
    measurement b { std::move(a) };

    b.measure();
    EXPECT_EQ(*count, 1);
}

TEST(MCMeasurement, CopyAssignReplacesPreviousValue) {
    counter_meas first;
    doubling_meas second;
    auto first_count = first.count;
    auto second_count = second.count;

    measurement m { first };
    m = measurement { second }; // assign a wrapper holding a different concrete type

    m.measure();

    EXPECT_EQ(*first_count, 0); // never reached
    EXPECT_EQ(*second_count, 2); // doubling_meas: 1 -> 2
}

TEST(MCMeasurement, HeterogeneousStorageInVector) {
    counter_meas a;
    doubling_meas b;
    auto a_count = a.count;
    auto b_count = b.count;

    std::vector<measurement> v;
    v.emplace_back(a);
    v.emplace_back(b);

    for (auto& m : v) {
        m.measure();
    }

    EXPECT_EQ(*a_count, 1); // counter_meas: 0 -> 1
    EXPECT_EQ(*b_count, 2); // doubling_meas: 1 -> 2
}
