#include "./mc_test_utils.hpp"

#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>

using namespace simplemc;

namespace {

// An update with only the required operations — no reject().
struct minimal_update {
    int accepts = 0;
    double attempt() { return 1.0; }
    void accept() { ++accepts; }
};

// A second distinct conforming update.
struct doubling_update {
    int committed = 1;
    double attempt() { return 0.25; }
    void accept() { committed *= 2; }
};

// Negative cases for the concept.
struct missing_attempt {
    void accept() {}
};

struct missing_accept {
    double attempt() { return 0.0; }
};

struct nothing {};

// Drive the wrapper so its run counters read (nprops, naccs, nimps).
template <mc_update U>
void drive_counters(update<U>& u, int nprops, int naccs, int nimps) {
    for (int i = 0; i < nprops; ++i) {
        u.attempt();
    }
    for (int i = 0; i < naccs; ++i) {
        u.accept();
    }
    for (int i = 0; i < nimps; ++i) {
        u.mark_impossible();
    }
}

} // namespace

// Check mc_update concept.
static_assert(mc_update<dummy_update>);
static_assert(mc_update<minimal_update>);
static_assert(mc_update<doubling_update>);
static_assert(!mc_update<missing_attempt>);
static_assert(!mc_update<missing_accept>);
static_assert(!mc_update<nothing>);
static_assert(!mc_update<int>);

// The update<U> wrapper forwards attempt()/accept() so it is itself an mc_update.
static_assert(mc_update<update<dummy_update>>);

// Test basic behavior of the measurement wrapper and the wrapped type.
TEST(MCUpdate, WrapsAndForwardsAttemptAccept) {
    dummy_update src;
    src.prob = 0.75;

    update u { src, "u", 1.0 };
    EXPECT_DOUBLE_EQ(u.attempt(), 0.75);
    EXPECT_EQ(u.stats().nprops, 1u);

    u.accept();
    EXPECT_EQ(u.value().accepts, 1);
    EXPECT_EQ(u.stats().naccs, 1u);
}

TEST(MCUpdate, RejectIsNoOpWhenUserTypeOmitsIt) {
    update u { minimal_update {}, "u", 1.0 };
    EXPECT_DOUBLE_EQ(u.attempt(), 1.0);
    u.reject();
    EXPECT_EQ(u.value().accepts, 0);
}

TEST(MCUpdate, RejectForwardsWhenUserTypeProvidesIt) {
    update u { dummy_update {}, "u", 1.0 };
    u.reject();
    EXPECT_EQ(u.value().rejects, 1);
}

TEST(MCUpdate, CopyProducesIndependentValue) {
    update a { dummy_update {}, "u", 1.0 };
    update b { a };
    a.accept();
    EXPECT_EQ(a.value().accepts, 1);
    EXPECT_EQ(b.value().accepts, 0);
}

TEST(MCUpdate, MoveTransfersOwnership) {
    update a { dummy_update {}, "u", 1.0 };
    a.accept();
    update b { std::move(a) };
    b.accept();
    EXPECT_EQ(b.value().accepts, 2);
}

TEST(MCUpdate, ValueGetterExposesUserUpdate) {
    dummy_update src;
    src.prob = 0.42;
    update u { src, "u", 1.0 };
    src.prob = 0.0; // the wrapper holds its own copy, unaffected by the source

    static_assert(std::is_same_v<decltype(u)::value_type, dummy_update>);
    EXPECT_DOUBLE_EQ(u.value().prob, 0.42);
    u.value().prob = 0.99;
    EXPECT_DOUBLE_EQ(u.attempt(), 0.99);
}

// Test constructors and access of metadata.
TEST(MCUpdate, ConstructFromUserValue) {
    update u { dummy_update {}, "u1", 2.5 };
    EXPECT_EQ(u.name(), "u1");
    EXPECT_EQ(u.stats().name, "u1");
    EXPECT_EQ(u.stats().inv_name, "u1");
    EXPECT_DOUBLE_EQ(u.stats().weight, 2.5);
    EXPECT_DOUBLE_EQ(u.stats().ratio, 1.0);
    EXPECT_EQ(u.stats().nprops, 0u);
    EXPECT_EQ(u.stats().naccs, 0u);
}

TEST(MCUpdate, ConstructorRejectsNegativeWeight) {
    EXPECT_THROW((update { dummy_update {}, "u", -0.5 }), simplemc_exception);
}

TEST(MCUpdate, ConstructorRejectsEmptyName) {
    EXPECT_THROW((update { dummy_update {}, "", 1.0 }), simplemc_exception);
}

// Test setter routines.
TEST(MCUpdate, SetWeightRejectsNegative) {
    update u { dummy_update {}, "u", 1.0 };
    EXPECT_THROW(u.set_weight(-1.0), simplemc_exception);
    EXPECT_DOUBLE_EQ(u.stats().weight, 1.0); // unchanged after the failed set
    u.set_weight(3.5);
    EXPECT_DOUBLE_EQ(u.stats().weight, 3.5);
}

TEST(MCUpdate, SetInvNameRejectsEmpty) {
    update u { dummy_update {}, "u", 1.0 };
    EXPECT_THROW(u.set_inv_name(""), simplemc_exception);
    EXPECT_EQ(u.stats().inv_name, "u"); // unchanged after the failed set
    u.set_inv_name("u_inv");
    EXPECT_EQ(u.stats().inv_name, "u_inv");
}

// Test that the wrapper maintains the counters itself.
TEST(MCUpdate, AttemptAcceptMarkImpossibleBumpCounters) {
    update u { dummy_update {}, "u", 1.0 };

    u.attempt();
    EXPECT_EQ(u.stats().nprops, 1u);
    EXPECT_EQ(u.stats().naccs, 0u);
    EXPECT_EQ(u.stats().nimps, 0u);

    u.accept();
    EXPECT_EQ(u.stats().naccs, 1u);
    EXPECT_EQ(u.value().accepts, 1);

    u.mark_impossible();
    EXPECT_EQ(u.stats().nimps, 1u);

    // reject() does not touch any counter
    u.reject();
    EXPECT_EQ(u.stats().nprops, 1u);
    EXPECT_EQ(u.stats().naccs, 1u);
    EXPECT_EQ(u.stats().nimps, 1u);
}

// Test that the counters accumulate across multiple drive rounds.
TEST(MCUpdate, CountersAccumulateAcrossDrives) {
    update u { dummy_update {}, "u", 1.0 };

    drive_counters(u, 7, 3, 1);
    drive_counters(u, 4, 0, 0);
    EXPECT_EQ(u.stats().nprops, 11u);
    EXPECT_EQ(u.stats().naccs, 3u);
    EXPECT_EQ(u.stats().nimps, 1u);
}

// Test that resetting zeros the counters but leaves the metadata untouched.
TEST(MCUpdate, ResetCountersZerosCountersOnly) {
    update u { dummy_update {}, "u", 2.5 };
    u.set_ratio(0.5);

    drive_counters(u, 7, 3, 1);
    u.reset_counters();
    EXPECT_EQ(u.stats().nprops, 0u);
    EXPECT_EQ(u.stats().naccs, 0u);
    EXPECT_EQ(u.stats().nimps, 0u);
    EXPECT_EQ(u.stats().name, "u");
    EXPECT_DOUBLE_EQ(u.stats().weight, 2.5);
    EXPECT_DOUBLE_EQ(u.stats().ratio, 0.5);
}

// Test serialization and deserialization of the wrapper and its metadata.
TEST(MCUpdate, SerializationRoundTrip) {
    update u { dummy_update {}, "u", 2.0 };
    u.set_inv_name("u_inv");
    u.set_ratio(0.5);
    drive_counters(u, 11, 7, 1);

    json_serializer s;
    auto entry = s["entry"];
    simplemc_save(entry, u);

    update v { dummy_update {}, "tmp", 1.0 };
    const auto rentry = s["entry"];
    simplemc_load(rentry, v);

    EXPECT_EQ(v.stats().inv_name, "u_inv");
    EXPECT_DOUBLE_EQ(v.stats().weight, u.stats().weight);
    EXPECT_DOUBLE_EQ(v.stats().ratio, u.stats().ratio);
    EXPECT_EQ(v.stats().nprops, 11u);
    EXPECT_EQ(v.stats().naccs, 7u);
    EXPECT_EQ(v.stats().nimps, 1u);
}

TEST(MCUpdate, InputConfigRoundTripOnlyTouchesWeight) {
    update u { dummy_update {}, "u", 4.0 };
    u.set_ratio(0.5);
    drive_counters(u, 42, 0, 0);

    json_serializer s;
    auto entry = s["entry"];
    simplemc_save_input_config(entry, u);

    update v { dummy_update {}, "u", 1.0 };
    v.set_ratio(9.0);
    const auto rentry = s["entry"];
    simplemc_load_input_config(rentry, v);

    EXPECT_DOUBLE_EQ(v.stats().weight, u.stats().weight); // updated
    EXPECT_DOUBLE_EQ(v.stats().ratio, 9.0); // untouched
    EXPECT_EQ(v.stats().nprops, 0u); // untouched
}

// Test the stats() snapshot.
TEST(MCUpdate, StatsReflectsMetadataAndCounters) {
    update u { dummy_update {}, "u", 2.0 };
    u.set_inv_name("u_inv");
    u.set_ratio(0.5);
    drive_counters(u, 5, 2, 1);

    const update_stats& s = u.stats();
    EXPECT_EQ(s.name, "u");
    EXPECT_EQ(s.inv_name, "u_inv");
    EXPECT_DOUBLE_EQ(s.weight, 2.0);
    EXPECT_DOUBLE_EQ(s.ratio, 0.5);
    EXPECT_EQ(s.nprops, 5u);
    EXPECT_EQ(s.naccs, 2u);
    EXPECT_EQ(s.nimps, 1u);

    // the reference tracks the live metadata
    u.attempt();
    EXPECT_EQ(s.nprops, 6u);
}

// Test printing of update_stats.
TEST(MCUpdate, PrintUpdateStats) {
    update u { dummy_update {}, "u", 2.0 };
    u.set_inv_name("u_inv");
    drive_counters(u, 10, 4, 1);
    print(u.stats());
}
