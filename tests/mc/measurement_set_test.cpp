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
    measurement_set ms;
    ms.add({ counter_meas {}, "a" });
    ms.add({ counter_meas {}, "b", false });
    EXPECT_EQ(ms.size(), 2u);
    EXPECT_TRUE(ms[0].is_active);
    EXPECT_FALSE(ms[1].is_active);
}

TEST(MCMeasurementSet, AddDuplicateNameThrows) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a" });
    EXPECT_THROW(ms.add({ counter_meas {}, "a" }), simplemc_exception);
}

TEST(MCMeasurementSet, SetActive) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", true });
    ms.set_active("a", false);
    EXPECT_FALSE(ms[0].is_active);

    EXPECT_THROW(ms.set_active("missing", false), simplemc_exception);
}

TEST(MCMeasurementSet, FindReturnsIndexOrNullopt) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a" });
    EXPECT_EQ(ms.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_FALSE(ms.find("missing").has_value());
}

TEST(MCMeasurementSet, RefreshActiveBuildsCache) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", true });
    ms.add({ counter_meas {}, "b", false });
    ms.add({ counter_meas {}, "c", true });

    ms.refresh_active();
    const auto& active = ms.active_indices();
    ASSERT_EQ(active.size(), 2u);
    EXPECT_EQ(active[0], 0u);
    EXPECT_EQ(active[1], 2u);
}

TEST(MCMeasurementSet, RefreshActiveSeesUpdatedFlags) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", true });
    ms.refresh_active();
    EXPECT_EQ(ms.active_indices().size(), 1u);

    ms.set_active("a", false);
    EXPECT_EQ(ms.active_indices().size(), 1u); // cache stale until refresh
    ms.refresh_active();
    EXPECT_EQ(ms.active_indices().size(), 0u);
}

TEST(MCMeasurementSet, ClearActiveEmptiesCache) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", true });
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

    measurement_set ms;
    ms.add({ a, "a", true });
    ms.add({ b, "b", false });
    ms.add({ c, "c", true });
    ms.refresh_active();

    ms.measure_all();
    ms.measure_all();

    EXPECT_EQ(*ca, 2);
    EXPECT_EQ(*cb, 0);
    EXPECT_EQ(*cc, 2);
}

TEST(MCMeasurementSet, GetReturnsTypedPointer) {
    measurement_set ms;
    counter_meas src;
    auto count = src.count;
    ms.add({ src, "a", true });

    auto* p = ms.get<counter_meas>("a");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->count.get(), count.get());

    EXPECT_EQ(ms.get<other_meas>("a"), nullptr);
    EXPECT_EQ(ms.get<counter_meas>("missing"), nullptr);
}

TEST(MCMeasurementSet, SerializationRoundTrip) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", true });
    ms.add({ counter_meas {}, "b", false });

    mc_serializer s { json_serializer {} };
    auto entry = s["measurements"];
    simplemc_save(entry, ms);

    measurement_set v;
    v.add({ counter_meas {}, "a", false });
    v.add({ counter_meas {}, "b", true });
    const auto rentry = mc_serializer { s }["measurements"];
    simplemc_load(rentry, v);

    EXPECT_TRUE(v[0].is_active);
    EXPECT_FALSE(v[1].is_active);
}

TEST(MCMeasurementSet, InputConfigRoundTrip) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", false });

    mc_serializer s { json_serializer {} };
    auto entry = s["measurements"];
    simplemc_save_input_config(entry, ms);

    measurement_set v;
    v.add({ counter_meas {}, "a", true });
    const auto rentry = mc_serializer { s }["measurements"];
    simplemc_load_input_config(rentry, v);

    EXPECT_FALSE(v[0].is_active);
}

TEST(MCMeasurementSet, LoadThrowsOnMissingEntry) {
    measurement_set ms;
    ms.add({ counter_meas {}, "a", true });
    ms.add({ counter_meas {}, "b", true });

    mc_serializer s { json_serializer {} };
    auto entry = s["measurements"];
    measurement_set partial;
    partial.add({ counter_meas {}, "a", true });
    simplemc_save(entry, partial);

    const auto rentry = mc_serializer { s }["measurements"];
    EXPECT_THROW(simplemc_load(rentry, ms), simplemc_exception);
}
