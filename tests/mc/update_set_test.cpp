#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <array>
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
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 2.0 } };

    EXPECT_EQ(us.size(), 2u);
    EXPECT_FALSE(us.empty());
    EXPECT_EQ(us.get<0>().name, "a");
    EXPECT_EQ(us.get<1>().name, "b");
    EXPECT_DOUBLE_EQ(us.get<0>().weight, 1.0);
    EXPECT_DOUBLE_EQ(us.get<1>().weight, 2.0);
    EXPECT_EQ(us.get<0>().inv_name, "a"); // self-inverse default
    EXPECT_DOUBLE_EQ(us.get<0>().ratio, 1.0);
}

// duplicate-name detection removed: roster is now a compile-time tuple

TEST(MCUpdateSet, AddPairCrossLinksAndRebuildComputesRatios) {
    update_set us { update { toy_update {}, "f", 2.0 }, update { toy_update {}, "b", 3.0 } };
    us.link_pair("f", "b");

    EXPECT_EQ(us.size(), 2u);
    EXPECT_EQ(us.get<0>().name, "f");
    EXPECT_EQ(us.get<0>().inv_name, "b");
    EXPECT_EQ(us.get<1>().name, "b");
    EXPECT_EQ(us.get<1>().inv_name, "f");

    us.rebuild_distribution(); // derives the detailed-balance ratios from the current weights
    EXPECT_DOUBLE_EQ(us.get<0>().ratio, 3.0 / 2.0);
    EXPECT_DOUBLE_EQ(us.get<1>().ratio, 2.0 / 3.0);
}

TEST(MCUpdateSet, SetWeightThenRebuildRecomputesRatios) {
    update_set us { update { toy_update {}, "f", 2.0 }, update { toy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");
    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<0>().ratio, 2.0);
    EXPECT_DOUBLE_EQ(us.get<1>().ratio, 0.5);

    // A later weight change must not leave the ratios stale.
    us.set_weight("f", 1.0);
    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<0>().ratio, 4.0);
    EXPECT_DOUBLE_EQ(us.get<1>().ratio, 0.25);
}

TEST(MCUpdateSet, RebuildThrowsOnAsymmetricZeroPair) {
    update_set us { update { toy_update {}, "a", 1.0 }, // keeps a positive total weight
        update { toy_update {}, "f", 2.0 }, update { toy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");

    us.set_weight("b", 0.0); // breaks the pair-zero invariant after registration
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, RebuildKeepsUnitRatioOnBothZeroPair) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "f", 0.0 },
        update { toy_update {}, "b", 0.0 } };
    us.link_pair("f", "b");

    us.rebuild_distribution();
    EXPECT_DOUBLE_EQ(us.get<1>().ratio, 1.0);
    EXPECT_DOUBLE_EQ(us.get<2>().ratio, 1.0);
}

TEST(MCUpdateSet, PairBothZeroAllowed) {
    update_set us { update { toy_update {}, "f", 0.0 }, update { toy_update {}, "b", 0.0 } };
    us.link_pair("f", "b");
    EXPECT_DOUBLE_EQ(us.get<0>().ratio, 1.0);
    EXPECT_DOUBLE_EQ(us.get<1>().ratio, 1.0);
}

TEST(MCUpdateSet, RebuildThrowsOnPairOneZero) {
    update_set us { update { toy_update {}, "f", 0.0 }, update { toy_update {}, "b", 1.0 } };
    us.link_pair("f", "b");
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, LinkPairSameNameThrows) {
    update_set us { update { toy_update {}, "x", 1.0 } };
    EXPECT_THROW(us.link_pair("x", "x"), simplemc_exception);
}

TEST(MCUpdateSet, LinkPairUnknownNameThrows) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 1.0 } };
    EXPECT_THROW(us.link_pair("a", "missing"), simplemc_exception);
}

TEST(MCUpdateSet, SetWeight) {
    update_set us { update { toy_update {}, "a", 1.0 } };
    us.set_weight("a", 5.5);
    EXPECT_DOUBLE_EQ(us.get<0>().weight, 5.5);

    EXPECT_THROW(us.set_weight("a", -1.0), simplemc_exception);
    EXPECT_THROW(us.set_weight("missing", 1.0), simplemc_exception);
}

TEST(MCUpdateSet, FindReturnsIndexOrNullopt) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 1.0 } };
    EXPECT_EQ(us.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_EQ(us.find("b"), std::optional<std::size_t> { 1 });
    EXPECT_FALSE(us.find("c").has_value());
}

TEST(MCUpdateSet, RebuildDistributionThrowsIfAllZero) {
    update_set us { update { toy_update {}, "a", 0.0 } };
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, SelectReturnsValidIndex) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 1.0 } };
    us.rebuild_distribution();

    std::mt19937_64 rng { 12345 };
    for (int i = 0; i < 64; ++i) {
        const std::size_t idx = us.select(rng);
        EXPECT_LT(idx, 2u);
    }
}

TEST(MCUpdateSet, SelectFrequenciesTrackWeights) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 2.0 },
        update { toy_update {}, "c", 4.0 } };
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

TEST(MCUpdateSet, RebuildThrowsOnDirectNegativeWeight) {
    update_set us { update { toy_update {}, "a", 1.0 } };
    us.get<0>().weight = -1.0; // public field: bypasses the constructor / set_weight validation
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, RebuildThrowsOnAsymmetricInversePairing) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "f", 2.0 },
        update { toy_update {}, "b", 3.0 } };
    us.link_pair("f", "b");

    us.get<2>().inv_name = "b"; // public field: hand-break the symmetry ('f' still names 'b')
    EXPECT_THROW(us.rebuild_distribution(), simplemc_exception);
}

TEST(MCUpdateSet, ResetAndAccumulateCounters) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 1.0 } };
    us.get<0>().nprops = 10;
    us.get<0>().naccs = 6;
    us.get<1>().nprops = 4;

    us.accumulate_counters();
    EXPECT_EQ(us.get<0>().cumulative_nprops, 10u);
    EXPECT_EQ(us.get<0>().cumulative_naccs, 6u);
    EXPECT_EQ(us.get<1>().cumulative_nprops, 4u);
    EXPECT_EQ(us.get<0>().nprops, 0u);
    EXPECT_EQ(us.get<1>().nprops, 0u);

    us.get<0>().nprops = 2;
    us.reset_run_counters();
    EXPECT_EQ(us.get<0>().nprops, 0u);
    EXPECT_EQ(us.get<0>().cumulative_nprops, 10u); // untouched
}

TEST(MCUpdateSet, GetReturnsTypedPointer) {
    toy_update src;
    src.prob = 0.42;
    update_set us { update { src, "a", 1.0 } };

    auto* p = us.get<toy_update>("a");
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->prob, 0.42);

    EXPECT_EQ(us.get<other_update>("a"), nullptr);
    EXPECT_EQ(us.get<toy_update>("missing"), nullptr);
}

TEST(MCUpdateSet, SerializationRoundTrip) {
    update_set us { update { toy_update {}, "a", 2.0 }, update { toy_update {}, "f", 3.0 },
        update { toy_update {}, "b", 4.0 } };
    us.link_pair("f", "b");
    us.get<0>().cumulative_nprops = 100;
    us.get<1>().cumulative_naccs = 50;

    json_serializer s;
    auto entry = s["updates"];
    simplemc_save(entry, us);

    update_set v { update { toy_update {}, "a", 0.0 }, update { toy_update {}, "f", 0.0 },
        update { toy_update {}, "b", 0.0 } };
    v.link_pair("f", "b");

    const auto rentry = s["updates"];
    simplemc_load(rentry, v);

    EXPECT_DOUBLE_EQ(v.get<0>().weight, 2.0);
    EXPECT_DOUBLE_EQ(v.get<1>().weight, 3.0);
    EXPECT_DOUBLE_EQ(v.get<2>().weight, 4.0);
    EXPECT_EQ(v.get<1>().inv_name, "b");
    EXPECT_EQ(v.get<0>().cumulative_nprops, 100u);
    EXPECT_EQ(v.get<1>().cumulative_naccs, 50u);
}

TEST(MCUpdateSet, InputConfigRoundTrip) {
    update_set us { update { toy_update {}, "a", 2.5 } };

    json_serializer s;
    auto entry = s["updates"];
    simplemc_save_input_config(entry, us);

    update_set v { update { toy_update {}, "a", 1.0 } };
    const auto rentry = s["updates"];
    simplemc_load_input_config(rentry, v);

    EXPECT_DOUBLE_EQ(v.get<0>().weight, 2.5);
}

TEST(MCUpdateSet, LoadThrowsOnMissingEntry) {
    update_set us { update { toy_update {}, "a", 1.0 }, update { toy_update {}, "b", 1.0 } };

    json_serializer s;
    auto entry = s["updates"];
    // serialize only one
    update_set partial { update { toy_update {}, "a", 1.0 } };
    simplemc_save(entry, partial);

    const auto rentry = s["updates"];
    EXPECT_THROW(simplemc_load(rentry, us), simplemc_exception);
}
