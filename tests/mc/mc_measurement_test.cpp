#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <utility>

using namespace simplemc;

namespace {

// A toy measurement that increments a counter so we can observe `measure()` calls through the
// wrapper's public `value` member.
struct counter_meas {
    int count = 0;
    void measure() { ++count; }
};

// A second, distinct conforming measurement.
struct doubling_meas {
    int count = 1;
    void measure() { count *= 2; }
};

// Negative case for the concept: missing `measure()`.
struct not_a_meas {
    int x = 0;
};

// A move-only measurement.
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

// Test basic behavior of the measurement wrapper and the wrapped type.
TEST(MCMeasurement, WrapsAndForwardsMeasure) {
    measurement m { counter_meas {}, "m" };
    m.measure();
    EXPECT_EQ(m.value.count, 1);
}

TEST(MCMeasurement, CopyProducesIndependentValue) {
    measurement a { counter_meas {}, "m" };
    measurement b { a }; // copy: b holds its own counter_meas with its own counter

    a.measure();
    EXPECT_EQ(a.value.count, 1);
    EXPECT_EQ(b.value.count, 0);
}

TEST(MCMeasurement, MoveTransfersOwnership) {
    measurement a { counter_meas {}, "m" };
    a.measure();

    measurement b { std::move(a) };
    b.measure();

    EXPECT_EQ(b.value.count, 2);
}

TEST(MCMeasurement, MoveOnlyMeasurementIsSupported) {
    measurement m { move_only_meas {}, "m" };
    m.measure();
    EXPECT_EQ(*m.value.count, 1);
}

TEST(MCMeasurement, ValueMemberExposesUserMeasurement) {
    counter_meas src;
    src.count = 5;
    measurement m { src, "m" };

    static_assert(std::is_same_v<decltype(m)::value_type, counter_meas>);
    EXPECT_EQ(m.value.count, 5);

    m.measure();
    EXPECT_EQ(m.value.count, 6);
}

// Test constructors and access of metadata.
TEST(MCMeasurement, ConstructFromUserValue) {
    measurement m { counter_meas {}, "m1" };
    EXPECT_EQ(m.name, "m1");
    EXPECT_TRUE(m.is_active);
}

TEST(MCMeasurement, ConstructFromUserValueNonActive) {
    measurement m { counter_meas {}, "m1", false };
    EXPECT_FALSE(m.is_active);
}

TEST(MCMeasurement, ConstructorRejectsEmptyName) {
    EXPECT_THROW((measurement { counter_meas {}, "", true }), simplemc_exception);
}

// Test serialization and deserialization of the wrapper and its metadata.
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
