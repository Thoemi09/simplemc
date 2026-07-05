#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>

using namespace simplemc;

namespace {

struct counter_meas {
    int count = 0;
    void measure() { ++count; }
};

struct other_meas {
    void measure() {}
};

} // namespace

// Test constructor.
TEST(MCMeasurementSet, AddRegistersEntries) {
    measurement_set ms { measurement { counter_meas {}, "a" }, measurement { counter_meas {}, "b", false } };
    EXPECT_EQ(ms.size(), 2u);
    EXPECT_TRUE(ms.get<0>().is_active);
    EXPECT_FALSE(ms.get<1>().is_active);
}

// Test changing the active state of a registered measurement.
TEST(MCMeasurementSet, SetActive) {
    measurement_set ms { measurement { counter_meas {}, "a", true } };
    EXPECT_TRUE(ms.get<0>().is_active);
    ms.set_active("a", false);
    EXPECT_FALSE(ms.get<0>().is_active);

    // throw on missing name
    EXPECT_THROW(ms.set_active("missing", false), simplemc_exception);
}

// Test finding a (non-)registered measurement by name.
TEST(MCMeasurementSet, FindReturnsIndexOrNullopt) {
    measurement_set ms { measurement { counter_meas {}, "a" } };
    EXPECT_EQ(ms.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_FALSE(ms.find("missing").has_value());
}

// Test rebuilding and clearing the active set cache.
TEST(MCMeasurementSet, RefreshActiveBuildsCache) {
    measurement_set ms { measurement { counter_meas {}, "a", true }, measurement { counter_meas {}, "b", false },
        measurement { counter_meas {}, "c", true } };

    ms.refresh_active();
    const auto& active = ms.active_indices();
    ASSERT_EQ(active.size(), 2u);
    EXPECT_EQ(active[0], 0u);
    EXPECT_EQ(active[1], 2u);
}

TEST(MCMeasurementSet, RefreshActiveSeesUpdatedFlags) {
    measurement_set ms { measurement { counter_meas {}, "a", true } };
    ms.refresh_active();
    EXPECT_EQ(ms.active_indices().size(), 1u);

    ms.set_active("a", false);
    EXPECT_EQ(ms.active_indices().size(), 1u); // cache stale until refresh
    ms.refresh_active();
    EXPECT_EQ(ms.active_indices().size(), 0u);
}

TEST(MCMeasurementSet, ClearActiveEmptiesCache) {
    measurement_set ms { measurement { counter_meas {}, "a", true } };
    ms.refresh_active();
    EXPECT_EQ(ms.active_indices().size(), 1u);
    ms.clear_active();
    EXPECT_EQ(ms.active_indices().size(), 0u);
}

// Test that measure_all() invokes measure() on every active entry.
TEST(MCMeasurementSet, MeasureAllInvokesActiveOnly) {
    measurement_set ms { measurement { counter_meas {}, "a", true }, measurement { counter_meas {}, "b", false },
        measurement { counter_meas {}, "c", true } };
    ms.refresh_active();

    ms.measure_all();
    ms.measure_all();

    EXPECT_EQ(ms.get<0>().value.count, 2);
    EXPECT_EQ(ms.get<1>().value.count, 0);
    EXPECT_EQ(ms.get<2>().value.count, 2);
}

// Test that get<T>(name) returns a typed pointer to the user measurement, or nullptr if the name is
// not registered or the type does not match.
TEST(MCMeasurementSet, GetReturnsTypedPointer) {
    counter_meas src;
    src.count = 7;
    measurement_set ms { measurement { src, "a", true } };

    auto* p = ms.get<counter_meas>("a");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->count, 7);

    EXPECT_EQ(ms.get<other_meas>("a"), nullptr);
    EXPECT_EQ(ms.get<counter_meas>("missing"), nullptr);
}

// Test serialization and deserialization of the measurement_set and its metadata.
TEST(MCMeasurementSet, SerializationRoundTrip) {
    measurement_set ms { measurement { counter_meas {}, "a", true }, measurement { counter_meas {}, "b", false } };

    json_serializer s;
    auto entry = s["measurements"];
    simplemc_save(entry, ms);

    measurement_set v { measurement { counter_meas {}, "a", false }, measurement { counter_meas {}, "b", true } };
    const auto rentry = s["measurements"];
    simplemc_load(rentry, v);

    EXPECT_TRUE(v.get<0>().is_active);
    EXPECT_FALSE(v.get<1>().is_active);
}

TEST(MCMeasurementSet, InputConfigRoundTrip) {
    measurement_set ms { measurement { counter_meas {}, "a", false } };

    json_serializer s;
    auto entry = s["measurements"];
    simplemc_save_input_config(entry, ms);

    measurement_set v { measurement { counter_meas {}, "a", true } };
    const auto rentry = s["measurements"];
    simplemc_load_input_config(rentry, v);

    EXPECT_FALSE(v.get<0>().is_active);
}

TEST(MCMeasurementSet, LoadThrowsOnMissingEntry) {
    measurement_set ms { measurement { counter_meas {}, "a", true }, measurement { counter_meas {}, "b", true } };

    json_serializer s;
    auto entry = s["measurements"];
    measurement_set partial { measurement { counter_meas {}, "a", true } };
    simplemc_save(entry, partial);

    const auto rentry = s["measurements"];
    EXPECT_THROW(simplemc_load(rentry, ms), simplemc_exception);
}
