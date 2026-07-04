#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <utility>

using namespace simplemc;

namespace {

// A toy measurement that increments a shared counter so we can observe `measure()` calls
// across copies.
struct counter_meas {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

// A second, distinct conforming type.
struct doubling_meas {
    std::shared_ptr<int> count = std::make_shared<int>(1);
    void measure() { *count *= 2; }
};

// Negative case for the concept: missing `measure()`.
struct not_a_meas {
    int x = 0;
};

// A move-only mc_measurement. Type erasure used to reject these (it required copyability); the
// tuple-based design stores the user type by value, so move-only measurements now work.
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
static_assert(mc_measurement<move_only_meas>);

// The measurement<M> wrapper forwards measure() so it is itself an mc_measurement.
static_assert(mc_measurement<measurement<counter_meas>>);

// =====================================================================================
// measurement<M> forwarding, value semantics and typed access.
// =====================================================================================

TEST(MCMeasurementForwarding, WrapsAndForwardsMeasure) {
    counter_meas src;
    auto count = src.count;

    measurement m { src, "m" };
    m.measure();

    EXPECT_EQ(*count, 1);
}

TEST(MCMeasurementForwarding, CopyProducesIndependentValueSharingCapturedState) {
    counter_meas src;
    auto count = src.count;

    measurement a { src, "m" };
    measurement b { a }; // copy: b holds its own counter_meas, still sharing the shared_ptr<int>

    a.measure();
    b.measure();

    EXPECT_EQ(*count, 2);
}

TEST(MCMeasurementForwarding, MoveTransfersOwnership) {
    counter_meas src;
    auto count = src.count;

    measurement a { src, "m" };
    measurement b { std::move(a) };

    b.measure();
    EXPECT_EQ(*count, 1);
}

TEST(MCMeasurementForwarding, MoveOnlyPayloadIsSupported) {
    measurement m { move_only_meas {}, "m" };
    m.measure();
    EXPECT_EQ(*m.get<move_only_meas>()->count, 1);
}

TEST(MCMeasurementForwarding, GetReturnsWrappedValueWhenTypeMatches) {
    counter_meas src;
    auto count = src.count;
    measurement m { src, "m" };

    auto* p = m.get<counter_meas>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->count.get(), count.get());

    // measuring through the wrapper bumps the same counter the recovered pointer sees
    m.measure();
    EXPECT_EQ(*p->count, 1);
}

TEST(MCMeasurementForwarding, GetReturnsNullOnTypeMismatch) {
    measurement m { counter_meas {}, "m" };
    EXPECT_EQ(m.get<doubling_meas>(), nullptr);
    EXPECT_EQ(m.get<int>(), nullptr);
}

TEST(MCMeasurementForwarding, GetConstOverloadReturnsConstPointer) {
    const measurement m { counter_meas {}, "m" };
    const counter_meas* p = m.get<counter_meas>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p->count, 0);
}

// =====================================================================================
// measurement<M> metadata: construction, validation, serialization.
// =====================================================================================

TEST(MCMeasurement, ConstructFromUserValue) {
    measurement m { counter_meas {}, "m1" };
    EXPECT_EQ(m.name, "m1");
    EXPECT_TRUE(m.is_active);
}

TEST(MCMeasurement, ConstructFromUserValueExplicitActive) {
    measurement m { counter_meas {}, "m1", false };
    EXPECT_FALSE(m.is_active);
}

TEST(MCMeasurement, ConstructorRejectsEmptyName) {
    EXPECT_THROW((measurement { counter_meas {}, "", true }), simplemc_exception);
}

TEST(MCMeasurement, MeasureForwardsToPayload) {
    counter_meas src;
    auto count = src.count;
    measurement m { src, "m", true };
    m.measure();
    EXPECT_EQ(*count, 1);
}

TEST(MCMeasurement, GetForwardsToPayload) {
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

    json_serializer s;
    auto entry = s["entry"];
    simplemc_save(entry, m);

    measurement v { counter_meas {}, "tmp", true };
    const auto rentry = s["entry"];
    simplemc_load(rentry, v);

    EXPECT_FALSE(v.is_active);
}

TEST(MCMeasurement, InputConfigRoundTripTouchesActive) {
    measurement m { counter_meas {}, "m", false };

    json_serializer s;
    auto entry = s["entry"];
    simplemc_save_input_config(entry, m);

    measurement v { counter_meas {}, "m", true };
    const auto rentry = s["entry"];
    simplemc_load_input_config(rentry, v);

    EXPECT_FALSE(v.is_active);
}
