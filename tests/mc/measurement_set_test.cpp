#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <memory>

using namespace simplemc;

namespace {

struct counter_meas {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

struct other_meas {
    void measure() {}
};

} // namespace

TEST(MCMeasurementSet, AddRegistersEntries) {
    measurement_set ms { measurement { counter_meas {}, "a" }, measurement { counter_meas {}, "b", false } };
    EXPECT_EQ(ms.size(), 2u);
    EXPECT_TRUE(ms.at<0>().is_active);
    EXPECT_FALSE(ms.at<1>().is_active);
}

// duplicate-name detection removed: roster is now a compile-time tuple

TEST(MCMeasurementSet, SetActive) {
    measurement_set ms { measurement { counter_meas {}, "a", true } };
    ms.set_active("a", false);
    EXPECT_FALSE(ms.at<0>().is_active);

    EXPECT_THROW(ms.set_active("missing", false), simplemc_exception);
}

TEST(MCMeasurementSet, FindReturnsIndexOrNullopt) {
    measurement_set ms { measurement { counter_meas {}, "a" } };
    EXPECT_EQ(ms.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_FALSE(ms.find("missing").has_value());
}

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

TEST(MCMeasurementSet, MeasureAllInvokesActiveOnly) {
    counter_meas a;
    counter_meas b;
    counter_meas c;
    auto ca = a.count;
    auto cb = b.count;
    auto cc = c.count;

    measurement_set ms { measurement { a, "a", true }, measurement { b, "b", false },
        measurement { c, "c", true } };
    ms.refresh_active();

    ms.measure_all();
    ms.measure_all();

    EXPECT_EQ(*ca, 2);
    EXPECT_EQ(*cb, 0);
    EXPECT_EQ(*cc, 2);
}

TEST(MCMeasurementSet, GetReturnsTypedPointer) {
    counter_meas src;
    auto count = src.count;
    measurement_set ms { measurement { src, "a", true } };

    auto* p = ms.get<counter_meas>("a");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->count.get(), count.get());

    EXPECT_EQ(ms.get<other_meas>("a"), nullptr);
    EXPECT_EQ(ms.get<counter_meas>("missing"), nullptr);
}

TEST(MCMeasurementSet, SerializationRoundTrip) {
    measurement_set ms { measurement { counter_meas {}, "a", true }, measurement { counter_meas {}, "b", false } };

    json_serializer s;
    auto entry = s["measurements"];
    simplemc_save(entry, ms);

    measurement_set v { measurement { counter_meas {}, "a", false }, measurement { counter_meas {}, "b", true } };
    const auto rentry = s["measurements"];
    simplemc_load(rentry, v);

    EXPECT_TRUE(v.at<0>().is_active);
    EXPECT_FALSE(v.at<1>().is_active);
}

TEST(MCMeasurementSet, InputConfigRoundTrip) {
    measurement_set ms { measurement { counter_meas {}, "a", false } };

    json_serializer s;
    auto entry = s["measurements"];
    simplemc_save_input_config(entry, ms);

    measurement_set v { measurement { counter_meas {}, "a", true } };
    const auto rentry = s["measurements"];
    simplemc_load_input_config(rentry, v);

    EXPECT_FALSE(v.at<0>().is_active);
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
