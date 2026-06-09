#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <random>

using namespace simplemc;

namespace {

struct toy_update {
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    double prob = 0.5;
    double attempt() { return prob; }
    void accept() { ++*accepts; }
    void reject() {}
};

struct other_update {
    double attempt() { return 1.0; }
    void accept() {}
};

} // namespace

TEST(MCUpdateSet, AddRegistersEntries) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    us.add({ toy_update {}, "b", 2.0 });

    EXPECT_EQ(us.size(), 2u);
    EXPECT_FALSE(us.empty());
    EXPECT_EQ(us.at(0).name, "a");
    EXPECT_EQ(us.at(1).name, "b");
    EXPECT_DOUBLE_EQ(us.at(0).weight, 1.0);
    EXPECT_DOUBLE_EQ(us.at(1).weight, 2.0);
    EXPECT_EQ(us.at(0).inv_name, "a"); // self-inverse default
    EXPECT_DOUBLE_EQ(us.at(0).ratio, 1.0);
}

TEST(MCUpdateSet, AddDuplicateNameThrows) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    EXPECT_THROW(us.add({ toy_update {}, "a", 1.0 }), simplemc_exception);
}

TEST(MCUpdateSet, AddPairCrossLinksAndComputesRatios) {
    update_set<> us;
    us.add_pair({ toy_update {}, "f", 2.0 }, { toy_update {}, "b", 3.0 });

    EXPECT_EQ(us.size(), 2u);
    EXPECT_EQ(us.at(0).name, "f");
    EXPECT_EQ(us.at(0).inv_name, "b");
    EXPECT_DOUBLE_EQ(us.at(0).ratio, 3.0 / 2.0);
    EXPECT_EQ(us.at(1).name, "b");
    EXPECT_EQ(us.at(1).inv_name, "f");
    EXPECT_DOUBLE_EQ(us.at(1).ratio, 2.0 / 3.0);
}

TEST(MCUpdateSet, AddPairBothZeroAllowed) {
    update_set<> us;
    EXPECT_NO_THROW(us.add_pair({ toy_update {}, "f", 0.0 }, { toy_update {}, "b", 0.0 }));
    EXPECT_DOUBLE_EQ(us.at(0).ratio, 1.0);
    EXPECT_DOUBLE_EQ(us.at(1).ratio, 1.0);
}

TEST(MCUpdateSet, AddPairOneZeroThrows) {
    update_set<> us;
    EXPECT_THROW(us.add_pair({ toy_update {}, "f", 0.0 }, { toy_update {}, "b", 1.0 }), simplemc_exception);
}

TEST(MCUpdateSet, AddPairSameNameThrows) {
    update_set<> us;
    EXPECT_THROW(us.add_pair({ toy_update {}, "x", 1.0 }, { toy_update {}, "x", 1.0 }), simplemc_exception);
}

TEST(MCUpdateSet, AddPairExistingNameThrows) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    EXPECT_THROW(us.add_pair({ toy_update {}, "a", 1.0 }, { toy_update {}, "b", 1.0 }), simplemc_exception);
}

TEST(MCUpdateSet, SetWeight) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    us.set_weight("a", 5.5);
    EXPECT_DOUBLE_EQ(us.at(0).weight, 5.5);

    EXPECT_THROW(us.set_weight("a", -1.0), simplemc_exception);
    EXPECT_THROW(us.set_weight("missing", 1.0), simplemc_exception);
}

TEST(MCUpdateSet, FindReturnsIndexOrNullopt) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    us.add({ toy_update {}, "b", 1.0 });
    EXPECT_EQ(us.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_EQ(us.find("b"), std::optional<std::size_t> { 1 });
    EXPECT_FALSE(us.find("c").has_value());
}

TEST(MCUpdateSet, RebuildDistributionThrowsIfAllZero) {
    update_set<> us;
    us.add({ toy_update {}, "a", 0.0 });
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, SelectReturnsValidIndex) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    us.add({ toy_update {}, "b", 1.0 });
    us.rebuild_distribution();

    std::mt19937_64 rng { 12345 };
    for (int i = 0; i < 64; ++i) {
        const std::size_t idx = us.select(rng);
        EXPECT_LT(idx, 2u);
    }
}

TEST(MCUpdateSet, ResetAndAccumulateCounters) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    us.add({ toy_update {}, "b", 1.0 });
    us.at(0).nprops = 10;
    us.at(0).naccs = 6;
    us.at(1).nprops = 4;

    us.accumulate_counters();
    EXPECT_EQ(us.at(0).cumulative_nprops, 10u);
    EXPECT_EQ(us.at(0).cumulative_naccs, 6u);
    EXPECT_EQ(us.at(1).cumulative_nprops, 4u);
    EXPECT_EQ(us.at(0).nprops, 0u);
    EXPECT_EQ(us.at(1).nprops, 0u);

    us.at(0).nprops = 2;
    us.reset_run_counters();
    EXPECT_EQ(us.at(0).nprops, 0u);
    EXPECT_EQ(us.at(0).cumulative_nprops, 10u); // untouched
}

TEST(MCUpdateSet, GetReturnsTypedPointer) {
    update_set<> us;
    toy_update src;
    src.prob = 0.42;
    us.add({ src, "a", 1.0 });

    auto* p = us.get<toy_update>("a");
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->prob, 0.42);

    EXPECT_EQ(us.get<other_update>("a"), nullptr);
    EXPECT_EQ(us.get<toy_update>("missing"), nullptr);
}

TEST(MCUpdateSet, SerializationRoundTrip) {
    update_set<> us;
    us.add({ toy_update {}, "a", 2.0 });
    us.add_pair({ toy_update {}, "f", 3.0 }, { toy_update {}, "b", 4.0 });
    us.at(0).cumulative_nprops = 100;
    us.at(1).cumulative_naccs = 50;

    json_serializer s;
    auto entry = s["updates"];
    simplemc_save(entry, us);

    update_set<> v;
    v.add({ toy_update {}, "a", 0.0 });
    v.add_pair({ toy_update {}, "f", 0.0 }, { toy_update {}, "b", 0.0 });

    const auto rentry = json_serializer { s }["updates"];
    simplemc_load(rentry, v);

    EXPECT_DOUBLE_EQ(v.at(0).weight, 2.0);
    EXPECT_DOUBLE_EQ(v.at(1).weight, 3.0);
    EXPECT_DOUBLE_EQ(v.at(2).weight, 4.0);
    EXPECT_EQ(v.at(1).inv_name, "b");
    EXPECT_EQ(v.at(0).cumulative_nprops, 100u);
    EXPECT_EQ(v.at(1).cumulative_naccs, 50u);
}

TEST(MCUpdateSet, InputConfigRoundTrip) {
    update_set<> us;
    us.add({ toy_update {}, "a", 2.5 });

    json_serializer s;
    auto entry = s["updates"];
    simplemc_save_input_config(entry, us);

    update_set<> v;
    v.add({ toy_update {}, "a", 1.0 });
    const auto rentry = json_serializer { s }["updates"];
    simplemc_load_input_config(rentry, v);

    EXPECT_DOUBLE_EQ(v.at(0).weight, 2.5);
}

TEST(MCUpdateSet, LoadThrowsOnMissingEntry) {
    update_set<> us;
    us.add({ toy_update {}, "a", 1.0 });
    us.add({ toy_update {}, "b", 1.0 });

    json_serializer s;
    auto entry = s["updates"];
    // serialize only one
    update_set<> partial;
    partial.add({ toy_update {}, "a", 1.0 });
    simplemc_save(entry, partial);

    const auto rentry = json_serializer { s }["updates"];
    EXPECT_THROW(simplemc_load(rentry, us), simplemc_exception);
}
