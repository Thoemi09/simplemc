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
static_assert(mc_measurement<basic_measurement>);

// the concept is a pure role description: a move-only measurement still satisfies it
// (but the wrapper rejects it because of the copyability requirement)
static_assert(mc_measurement<move_only_meas>);
static_assert(!std::is_constructible_v<basic_measurement, int>);
static_assert(!std::is_constructible_v<basic_measurement, not_a_meas>);

TEST(MCBasicMeasurement, WrapsAndForwardsMeasure) {
    counter_meas src;
    auto count = src.count;

    basic_measurement m { src };
    m.measure();

    EXPECT_EQ(*count, 1);
}

TEST(MCBasicMeasurement, CopyProducesIndependentValueSharingCapturedState) {
    counter_meas src;
    auto count = src.count;

    basic_measurement a { src };
    basic_measurement b { a }; // deep copy via clone()

    a.measure();
    b.measure();

    // both wrappers' models hold their own copy of `counter_meas`
    EXPECT_EQ(*count, 2);
}

TEST(MCBasicMeasurement, MoveTransfersOwnership) {
    counter_meas src;
    auto count = src.count;

    basic_measurement a { src };
    basic_measurement b { std::move(a) };

    b.measure();
    EXPECT_EQ(*count, 1);
}

TEST(MCBasicMeasurement, CopyAssignReplacesPreviousValue) {
    counter_meas first;
    doubling_meas second;
    auto first_count = first.count;
    auto second_count = second.count;

    basic_measurement m { first };
    m = basic_measurement { second }; // assign a wrapper holding a different concrete type

    m.measure();

    EXPECT_EQ(*first_count, 0); // never reached
    EXPECT_EQ(*second_count, 2); // doubling_meas: 1 -> 2
}

TEST(MCBasicMeasurement, HeterogeneousStorageInVector) {
    counter_meas a;
    doubling_meas b;
    auto a_count = a.count;
    auto b_count = b.count;

    std::vector<basic_measurement> v;
    v.emplace_back(a);
    v.emplace_back(b);

    for (auto& m : v) {
        m.measure();
    }

    EXPECT_EQ(*a_count, 1); // counter_meas: 0 -> 1
    EXPECT_EQ(*b_count, 2); // doubling_meas: 1 -> 2
}

TEST(MCBasicMeasurement, GetReturnsWrappedValueWhenTypeMatches) {
    counter_meas src;
    auto count = src.count;
    basic_measurement m { src };

    auto* p = m.get<counter_meas>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->count.get(), count.get());

    // measuring through the wrapper bumps the same counter the recovered pointer sees
    m.measure();
    EXPECT_EQ(*p->count, 1);
}

TEST(MCBasicMeasurement, GetReturnsNullOnTypeMismatch) {
    basic_measurement m { counter_meas{} };
    EXPECT_EQ(m.get<doubling_meas>(), nullptr);
    EXPECT_EQ(m.get<int>(), nullptr);
}

TEST(MCBasicMeasurement, GetConstOverloadReturnsConstPointer) {
    const basic_measurement m { counter_meas{} };
    const counter_meas* p = m.get<counter_meas>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p->count, 0);
}

// =====================================================================================
// Tests for the new per-measurement struct (basic_measurement wrapper + name + active).
// =====================================================================================

#include <simplemc/serialize/json/json_serializer.hpp>

TEST(MCMeasurement, ConstructFromUserValue) {
    measurement m { counter_meas {}, "m1" };
    EXPECT_EQ(m.name, "m1");
    EXPECT_TRUE(m.is_active);
}

TEST(MCMeasurement, ConstructFromUserValueExplicitActive) {
    measurement m { counter_meas {}, "m1", false };
    EXPECT_FALSE(m.is_active);
}

TEST(MCMeasurement, ConstructFromPreBuiltWrapper) {
    basic_measurement bw { counter_meas {} };
    measurement m { std::move(bw), "m2", true };
    EXPECT_EQ(m.name, "m2");
    EXPECT_TRUE(m.is_active);
}

TEST(MCMeasurement, ConstructorRejectsEmptyName) {
    EXPECT_THROW(measurement(counter_meas {}, "", true), simplemc_exception);
}

TEST(MCMeasurement, MeasureForwardsToWrapper) {
    counter_meas src;
    auto count = src.count;
    measurement m { src, "m", true };
    m.measure();
    EXPECT_EQ(*count, 1);
}

TEST(MCMeasurement, GetForwardsToWrapper) {
    counter_meas src;
    auto count = src.count;
    measurement m { src, "m", true };

    auto* p = m.get<counter_meas>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->count.get(), count.get());

    EXPECT_EQ(m.get<doubling_meas>(), nullptr);
}

TEST(MCMeasurement, SerializationRoundTrip) {
    measurement m { counter_meas {}, "m", false };

    mc_serializer s { json_serializer {} };
    auto entry = s["entry"];
    simplemc_save(entry, m);

    measurement v { counter_meas {}, "tmp", true };
    const auto rentry = mc_serializer { s }["entry"];
    simplemc_load(rentry, v);

    EXPECT_FALSE(v.is_active);
}

TEST(MCMeasurement, InputConfigRoundTripTouchesActive) {
    measurement m { counter_meas {}, "m", false };

    mc_serializer s { json_serializer {} };
    auto entry = s["entry"];
    simplemc_save_input_config(entry, m);

    measurement v { counter_meas {}, "m", true };
    const auto rentry = mc_serializer { s }["entry"];
    simplemc_load_input_config(rentry, v);

    EXPECT_FALSE(v.is_active);
}
