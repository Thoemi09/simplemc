#include "./mc_test_utils.hpp"

#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace simplemc;

namespace {

struct other_update {
    double attempt() { return 1.0; }
    void accept() {}
};

} // namespace

// Test constructor.
TEST(MCUpdateSet, ConstructorRegistersEntries) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 2.0 } };
    EXPECT_EQ(us.size(), 2u);
    EXPECT_FALSE(us.empty());
    EXPECT_EQ(us.get<0>().name(), "a");
    EXPECT_EQ(us.get<1>().name(), "b");
    EXPECT_DOUBLE_EQ(us.get<0>().stats().weight, 1.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().weight, 2.0);
    EXPECT_EQ(us.get<0>().stats().inv_name, "a");
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 1.0);
}

TEST(MCUpdateSet, ConstructorThrowsOnDuplicateNames) {
    EXPECT_THROW((update_set { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "a", 2.0 } }),
        simplemc_exception);
}

// Test linking pairs and rebuilding the selection distribution.
TEST(MCUpdateSet, AddPairCrossLinksAndRebuildComputesRatios) {
    update_set us { update { dummy_update {}, "f", 2.0 }, update { dummy_update {}, "b", 3.0 } };
    us.link_pair("f", "b");

    EXPECT_EQ(us.size(), 2u);
    EXPECT_EQ(us.get<0>().name(), "f");
    EXPECT_EQ(us.get<0>().stats().inv_name, "b");
    EXPECT_EQ(us.get<1>().name(), "b");
    EXPECT_EQ(us.get<1>().stats().inv_name, "f");

    us.rebuild_distribution(); // derives the detailed-balance ratios from the current weights
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 3.0 / 2.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 2.0 / 3.0);
}

TEST(MCUpdateSet, SetWeightThenRebuildRecomputesRatios) {
    update_set us { update { dummy_update {}, "f", 2.0 }, update { dummy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");
    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 2.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 0.5);

    // a later weight change must not leave the ratios stale
    us.set_weight("f", 1.0);
    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 4.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 0.25);
}

TEST(MCUpdateSet, RebuildThrowsOnAsymmetricZeroPair) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "f", 2.0 },
        update { dummy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");
    us.set_weight("b", 0.0); // breaks the pair-zero invariant after registration
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, RebuildAllowsSymmetricZeroPair) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "f", 0.0 },
        update { dummy_update {}, "b", 0.0 } };
    us.link_pair("f", "b");
    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 1.0);
    EXPECT_DOUBLE_EQ(us.get<2>().stats().ratio, 1.0);
}

TEST(MCUpdateSet, RebuildThrowsOnPairOneZero) {
    update_set us { update { dummy_update {}, "f", 0.0 }, update { dummy_update {}, "b", 1.0 } };
    us.rebuild_distribution();
    us.link_pair("f", "b");
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, LinkPairSameNameDeclaresSelfInverse) {
    update_set us { update { dummy_update {}, "f", 2.0 }, update { dummy_update {}, "b", 3.0 } };
    us.link_pair("f", "b");
    us.link_pair("f", "f"); // reverts "f" to self-inverse
    us.link_pair("b", "b"); // the former partner must be reset explicitly

    EXPECT_EQ(us.get<0>().stats().inv_name, "f");
    EXPECT_EQ(us.get<1>().stats().inv_name, "b");

    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 1.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 1.0);
}

TEST(MCUpdateSet, RebuildThrowsOnAsymmetricPairAfterRelink) {
    update_set us { update { dummy_update {}, "f", 2.0 }, update { dummy_update {}, "b", 3.0 },
        update { dummy_update {}, "g", 4.0 } };
    us.link_pair("f", "b");
    us.link_pair("f", "g"); // leaves "b" pointing at "f"

    EXPECT_EQ(us.get<0>().stats().inv_name, "g");
    EXPECT_EQ(us.get<1>().stats().inv_name, "f");
    EXPECT_EQ(us.get<2>().stats().inv_name, "f");
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);

    // resetting the former partner repairs the set
    us.link_pair("b", "b");
    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 4.0 / 2.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 1.0);
    EXPECT_DOUBLE_EQ(us.get<2>().stats().ratio, 2.0 / 4.0);
}

TEST(MCUpdateSet, LinkPairUnknownNameThrows) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 1.0 } };
    EXPECT_THROW(us.link_pair("a", "missing"), simplemc_exception);
}

TEST(MCUpdateSet, RebuildDistributionThrowsIfAllZero) {
    update_set us { update { dummy_update {}, "a", 0.0 } };
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

// Test setting the weight of an update by name.
TEST(MCUpdateSet, SetWeight) {
    update_set us { update { dummy_update {}, "a", 1.0 } };
    us.set_weight("a", 5.5);
    EXPECT_DOUBLE_EQ(us.get<0>().stats().weight, 5.5);

    // throw on negative weight or missing name
    EXPECT_THROW(us.set_weight("a", -1.0), simplemc_exception);
    EXPECT_THROW(us.set_weight("missing", 1.0), simplemc_exception);
}

// Test finding a (non-)registered measurement by name.
TEST(MCUpdateSet, FindReturnsIndexOrNullopt) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 1.0 } };
    EXPECT_EQ(us.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_EQ(us.find("b"), std::optional<std::size_t> { 1 });
    EXPECT_FALSE(us.find("c").has_value());
}

// Test selecting a random update according to the cached discrete distribution.
TEST(MCUpdateSet, SelectReturnsValidIndex) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 1.0 } };
    us.rebuild_distribution();

    std::mt19937_64 rng { 12345 };
    for (int i = 0; i < 64; ++i) {
        const std::size_t idx = us.select(rng);
        EXPECT_LT(idx, 2u);
    }
}

TEST(MCUpdateSet, SelectFrequenciesTrackWeights) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 2.0 },
        update { dummy_update {}, "c", 4.0 } };
    us.rebuild_distribution();

    std::mt19937_64 rng { 42 };
    constexpr int n = 100'000;
    std::array<int, 3> counts {};
    for (int i = 0; i < n; ++i) {
        ++counts[us.select(rng)];
    }

    // Standard error per bin is ~0.002 at n = 100'000; a 0.02 tolerance is > 10 sigma.
    const std::array<double, 3> expected { 1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0 };
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(static_cast<double>(counts[k]) / n, expected[k], 0.02);
    }
}

// Test resetting the counters of all updates in the set.
TEST(MCUpdateSet, ResetCountersZerosEveryUpdate) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 1.0 } };
    for (int i = 0; i < 10; ++i) {
        us.get<0>().attempt();
    }
    for (int i = 0; i < 6; ++i) {
        us.get<0>().accept();
    }
    for (int i = 0; i < 4; ++i) {
        us.get<1>().attempt();
    }

    us.reset_counters();
    EXPECT_EQ(us.get<0>().stats().nprops, 0u);
    EXPECT_EQ(us.get<0>().stats().naccs, 0u);
    EXPECT_EQ(us.get<1>().stats().nprops, 0u);
    EXPECT_DOUBLE_EQ(us.get<0>().stats().weight, 1.0); // untouched
}

// Test that get<T>(name) returns a typed pointer to the user update, or nullptr if the name is
// not registered or the type does not match.
TEST(MCUpdateSet, GetReturnsTypedPointer) {
    dummy_update src;
    src.prob = 0.42;
    update_set us { update { src, "a", 1.0 } };

    auto* p = us.get<dummy_update>("a");
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->prob, 0.42);

    EXPECT_EQ(us.get<other_update>("a"), nullptr);
    EXPECT_EQ(us.get<dummy_update>("missing"), nullptr);
}

// Test serialization and deserialization of the update_set and its metadata.
TEST(MCUpdateSet, SerializationRoundTrip) {
    update_set us { update { dummy_update {}, "a", 2.0 }, update { dummy_update {}, "f", 3.0 },
        update { dummy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");
    for (int i = 0; i < 100; ++i) {
        us.get<0>().attempt();
    }
    for (int i = 0; i < 50; ++i) {
        us.get<1>().accept();
    }

    json_serializer s;
    auto entry = s["updates"];
    simplemc_save(entry, us);

    update_set v { update { dummy_update {}, "a", 0.0 }, update { dummy_update {}, "f", 0.0 },
        update { dummy_update {}, "b", 0.0 } };
    v.link_pair("f", "b");

    const auto rentry = s["updates"];
    simplemc_load(rentry, v);

    EXPECT_DOUBLE_EQ(v.get<0>().stats().weight, 2.0);
    EXPECT_DOUBLE_EQ(v.get<1>().stats().weight, 3.0);
    EXPECT_DOUBLE_EQ(v.get<2>().stats().weight, 4.0);
    EXPECT_EQ(v.get<1>().stats().inv_name, "b");
    EXPECT_EQ(v.get<0>().stats().nprops, 100u);
    EXPECT_EQ(v.get<1>().stats().naccs, 50u);
}

TEST(MCUpdateSet, InputConfigRoundTrip) {
    update_set us { update { dummy_update {}, "a", 2.5 } };

    json_serializer s;
    auto entry = s["updates"];
    simplemc_save_input_config(entry, us);

    update_set v { update { dummy_update {}, "a", 1.0 } };
    const auto rentry = s["updates"];
    simplemc_load_input_config(rentry, v);

    EXPECT_DOUBLE_EQ(v.get<0>().stats().weight, 2.5);
}

TEST(MCUpdateSet, LoadThrowsOnMissingEntry) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 1.0 } };

    json_serializer s;
    auto entry = s["updates"];
    update_set partial { update { dummy_update {}, "a", 1.0 } };
    simplemc_save(entry, partial);

    const auto rentry = s["updates"];
    EXPECT_THROW(simplemc_load(rentry, us), simplemc_exception);
}

// Test update statistic snapshots of the entire set.
TEST(MCUpdateSet, StatsSnapshotsAllUpdatesInOrder) {
    update_set us { update { dummy_update {}, "f", 2.0 }, update { dummy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");
    us.rebuild_distribution();
    for (int i = 0; i < 3; ++i) {
        us.get<0>().attempt();
    }
    us.get<0>().accept();

    const std::vector<update_stats> ss = us.stats();
    ASSERT_EQ(ss.size(), 2u);
    EXPECT_EQ(ss[0].name, "f");
    EXPECT_EQ(ss[0].inv_name, "b");
    EXPECT_DOUBLE_EQ(ss[0].weight, 2.0);
    EXPECT_DOUBLE_EQ(ss[0].ratio, 2.0);
    EXPECT_EQ(ss[0].nprops, 3u);
    EXPECT_EQ(ss[0].naccs, 1u);
    EXPECT_EQ(ss[1].name, "b");
    EXPECT_EQ(ss[1].nprops, 0u);

    // the snapshot is decoupled from the live counters
    us.get<0>().attempt();
    EXPECT_EQ(ss[0].nprops, 3u);
}

// Test printing of a vector of update_stats.
TEST(MCUpdateSet, PrintVectorOfUpdateStats) {
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 2.0 } };
    us.get<0>().attempt();
    us.get<0>().accept();
    print(us.stats());
}
