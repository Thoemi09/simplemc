#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace simplemc;

namespace {

// A toy update with attempt(), accept(), and reject().
struct toy_update {
    int accepted = 0;
    int rejected = 0;
    double prob = 0.5;
    double attempt() { return prob; }
    void accept() { ++accepted; }
    void reject() { ++rejected; }
};

// An update with only the required operations — no reject().
struct minimal_update {
    int accepted = 0;
    double attempt() { return 1.0; }
    void accept() { ++accepted; }
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

} // namespace

// Check mc_update concept.
static_assert(mc_update<toy_update>);
static_assert(mc_update<minimal_update>);
static_assert(mc_update<doubling_update>);
static_assert(!mc_update<missing_attempt>);
static_assert(!mc_update<missing_accept>);
static_assert(!mc_update<nothing>);
static_assert(!mc_update<int>);

// The update<U> wrapper forwards attempt()/accept() so it is itself an mc_update.
static_assert(mc_update<update<toy_update>>);

// Test basic behavior of the measurement wrapper and the wrapped type.
TEST(MCUpdate, WrapsAndForwardsAttemptAccept) {
    toy_update src;
    src.prob = 0.75;

    update u { src, "u", 1.0 };
    EXPECT_DOUBLE_EQ(u.attempt(), 0.75);

    u.accept();
    EXPECT_EQ(u.value.accepted, 1);
}

TEST(MCUpdate, RejectIsNoOpWhenUserTypeOmitsIt) {
    update u { minimal_update {}, "u", 1.0 };
    EXPECT_DOUBLE_EQ(u.attempt(), 1.0);

    // should be a no-op: minimal_update has no reject(); the wrapper's reject() must not crash
    // and must not affect any observable state
    u.reject();
    EXPECT_EQ(u.value.accepted, 0);
}

TEST(MCUpdate, RejectForwardsWhenUserTypeProvidesIt) {
    update u { toy_update {}, "u", 1.0 };
    u.reject();

    EXPECT_EQ(u.value.rejected, 1);
}

TEST(MCUpdate, CopyProducesIndependentValue) {
    update a { toy_update {}, "u", 1.0 };
    update b { a }; // copy: b holds its own toy_update with its own counters

    a.accept();
    EXPECT_EQ(a.value.accepted, 1);
    EXPECT_EQ(b.value.accepted, 0);
}

TEST(MCUpdate, MoveTransfersOwnership) {
    update a { toy_update {}, "u", 1.0 };
    a.accept();

    update b { std::move(a) };
    b.accept();

    EXPECT_EQ(b.value.accepted, 2);
}

TEST(MCUpdate, ValueMemberExposesUserUpdate) {
    toy_update src;
    src.prob = 0.42;
    update u { src, "u", 1.0 };
    src.prob = 0.0; // the wrapper holds its own copy, unaffected by the source

    static_assert(std::is_same_v<decltype(u)::value_type, toy_update>);
    EXPECT_DOUBLE_EQ(u.value.prob, 0.42);

    // mutate through the public member and observe via the wrapper
    u.value.prob = 0.99;
    EXPECT_DOUBLE_EQ(u.attempt(), 0.99);
}

// Test constructors and access of metadata.
TEST(MCUpdate, ConstructFromUserValue) {
    update u { toy_update {}, "u1", 2.5 };
    EXPECT_EQ(u.name, "u1");
    EXPECT_EQ(u.inv_name, "u1");
    EXPECT_DOUBLE_EQ(u.weight, 2.5);
    EXPECT_DOUBLE_EQ(u.ratio, 1.0);
    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.cumulative_naccs, 0u);
}

TEST(MCUpdate, ConstructorRejectsNegativeWeight) {
    EXPECT_THROW((update { toy_update {}, "u", -0.5 }), simplemc_exception);
}

TEST(MCUpdate, ConstructorRejectsEmptyName) {
    EXPECT_THROW((update { toy_update {}, "", 1.0 }), simplemc_exception);
}

TEST(MCUpdate, CountersResetAndAccumulate) {
    update u { toy_update {}, "u", 1.0 };
    u.nprops = 7;
    u.naccs = 3;
    u.nimps = 1;

    u.accumulate_counters();
    EXPECT_EQ(u.cumulative_nprops, 7u);
    EXPECT_EQ(u.cumulative_naccs, 3u);
    EXPECT_EQ(u.cumulative_nimps, 1u);
    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.naccs, 0u);
    EXPECT_EQ(u.nimps, 0u);

    u.nprops = 4;
    u.reset_run_counters();
    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.cumulative_nprops, 7u); // cumulative untouched by reset
}

// Test serialization and deserialization of the wrapper and its metadata.
TEST(MCUpdate, SerializationRoundTrip) {
    update u { toy_update {}, "u", 2.0 };
    u.inv_name = "u_inv";
    u.ratio = 0.5;
    u.cumulative_nprops = 11;
    u.cumulative_naccs = 7;
    u.cumulative_nimps = 1;

    json_serializer s;
    auto entry = s["entry"];
    simplemc_save(entry, u);

    update v { toy_update {}, "tmp", 1.0 };
    const auto rentry = s["entry"];
    simplemc_load(rentry, v);

    EXPECT_EQ(v.inv_name, "u_inv");
    EXPECT_DOUBLE_EQ(v.weight, 2.0);
    EXPECT_DOUBLE_EQ(v.ratio, 0.5);
    EXPECT_EQ(v.cumulative_nprops, 11u);
    EXPECT_EQ(v.cumulative_naccs, 7u);
    EXPECT_EQ(v.cumulative_nimps, 1u);
}

TEST(MCUpdate, InputConfigRoundTripOnlyTouchesWeight) {
    update u { toy_update {}, "u", 4.0 };
    u.ratio = 0.5;
    u.cumulative_nprops = 42;

    json_serializer s;
    auto entry = s["entry"];
    simplemc_save_input_config(entry, u);

    // loader picks up only "weight"; ratio and cumulative_* are untouched
    update v { toy_update {}, "u", 1.0 };
    v.ratio = 9.0;
    v.cumulative_nprops = 0;
    const auto rentry = s["entry"];
    simplemc_load_input_config(rentry, v);

    EXPECT_DOUBLE_EQ(v.weight, 4.0);
    EXPECT_DOUBLE_EQ(v.ratio, 9.0); // untouched
    EXPECT_EQ(v.cumulative_nprops, 0u); // untouched
}
