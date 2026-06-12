#include <simplemc/mc.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <vector>

using namespace simplemc;

namespace {

// A toy update with attempt(), accept(), and reject() — observable side effects via shared
// counters so we can see what got called.
struct toy_update {
    std::shared_ptr<int> accepted = std::make_shared<int>(0);
    std::shared_ptr<int> rejected = std::make_shared<int>(0);
    double prob = 0.5;
    double attempt() { return prob; }
    void accept() { ++*accepted; }
    void reject() { ++*rejected; }
};

// An update with only the required operations — no reject().
struct minimal_update {
    std::shared_ptr<int> accepted = std::make_shared<int>(0);
    double attempt() { return 1.0; }
    void accept() { ++*accepted; }
};

// A second distinct conforming type for heterogeneous-storage tests.
struct doubling_update {
    std::shared_ptr<int> committed = std::make_shared<int>(1);
    double attempt() { return 0.25; }
    void accept() { *committed *= 2; }
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

// The wrapper itself satisfies the concept — it forwards attempt() and accept().
static_assert(mc_update<basic_update>);

// A non-conforming type cannot construct an `update`.
static_assert(!std::is_constructible_v<basic_update, int>);
static_assert(!std::is_constructible_v<basic_update, nothing>);
static_assert(!std::is_constructible_v<basic_update, missing_attempt>);
static_assert(!std::is_constructible_v<basic_update, missing_accept>);

TEST(MCBasicUpdate, WrapsAndForwardsAttemptAccept) {
    toy_update src;
    src.prob = 0.75;
    auto accepted = src.accepted;

    basic_update u { src };

    EXPECT_DOUBLE_EQ(u.attempt(), 0.75);

    u.accept();
    EXPECT_EQ(*accepted, 1);
}

TEST(MCBasicUpdate, RejectIsNoOpWhenUserTypeOmitsIt) {
    minimal_update src;
    auto accepted = src.accepted;

    basic_update u { src };
    EXPECT_DOUBLE_EQ(u.attempt(), 1.0);

    // should be a no-op: minimal_update has no reject(); the wrapper's reject() must not crash
    // and must not affect any observable state
    u.reject();

    EXPECT_EQ(*accepted, 0);
}

TEST(MCBasicUpdate, RejectForwardsWhenUserTypeProvidesIt) {
    toy_update src;
    auto rejected = src.rejected;

    basic_update u { src };
    u.reject();

    EXPECT_EQ(*rejected, 1);
}

TEST(MCBasicUpdate, CopyProducesIndependentValueSharingCapturedState) {
    toy_update src;
    auto accepted = src.accepted;

    basic_update a { src };
    basic_update b { a }; // deep copy via clone()

    a.accept();
    b.accept();

    // both wrappers' models hold their own copy of toy_update, but each copy retains the same
    // shared_ptr<int>, so the deep copy is visible while the counter is shared
    EXPECT_EQ(*accepted, 2);
}

TEST(MCBasicUpdate, MoveTransfersOwnership) {
    toy_update src;
    auto accepted = src.accepted;

    basic_update a { src };
    basic_update b { std::move(a) };

    b.accept();
    EXPECT_EQ(*accepted, 1);
}

TEST(MCBasicUpdate, CopyAssignReplacesPreviousValue) {
    toy_update first;
    doubling_update second;
    auto first_accepted = first.accepted;
    auto second_committed = second.committed;

    basic_update u { first };
    u = basic_update { second }; // assign a wrapper holding a different concrete type

    u.accept();

    EXPECT_EQ(*first_accepted, 0); // never reached
    EXPECT_EQ(*second_committed, 2); // doubling_update: 1 -> 2
}

TEST(MCBasicUpdate, HeterogeneousStorageInVector) {
    toy_update a;
    doubling_update b;
    auto a_accepted = a.accepted;
    auto b_committed = b.committed;

    std::vector<basic_update> v;
    v.emplace_back(a);
    v.emplace_back(b);

    for (auto& u : v) {
        (void)u.attempt();
        u.accept();
    }

    EXPECT_EQ(*a_accepted, 1); // toy_update: 0 -> 1
    EXPECT_EQ(*b_committed, 2); // doubling_update: 1 -> 2
}

TEST(MCBasicUpdate, GetReturnsWrappedValueWhenTypeMatches) {
    toy_update src;
    src.prob = 0.42;
    basic_update u { src };

    auto* p = u.get<toy_update>();
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->prob, 0.42);

    // mutate through the recovered pointer and observe via the wrapper
    p->prob = 0.99;
    EXPECT_DOUBLE_EQ(u.attempt(), 0.99);
}

TEST(MCBasicUpdate, GetReturnsNullOnTypeMismatch) {
    basic_update u { toy_update{} };
    EXPECT_EQ(u.get<doubling_update>(), nullptr);
    EXPECT_EQ(u.get<int>(), nullptr);
}

TEST(MCBasicUpdate, GetConstOverloadReturnsConstPointer) {
    const basic_update u { toy_update { .prob = 0.5 } };
    const toy_update* p = u.get<toy_update>();
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->prob, 0.5);
}

// =====================================================================================
// Tests for the new per-update struct (basic_update wrapper + name + weight + counters).
// =====================================================================================

#include <simplemc/serialize/json/json_serializer.hpp>

TEST(MCUpdate, ConstructFromUserValue) {
    update u { toy_update {}, "u1", 2.5 };
    EXPECT_EQ(u.name, "u1");
    EXPECT_EQ(u.inv_name, "u1");
    EXPECT_DOUBLE_EQ(u.weight, 2.5);
    EXPECT_DOUBLE_EQ(u.ratio, 1.0);
    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.cumulative_naccs, 0u);
}

TEST(MCUpdate, ConstructFromPreBuiltWrapper) {
    basic_update bw { toy_update {} };
    update u { std::move(bw), "u2", 1.5 };
    EXPECT_EQ(u.name, "u2");
    EXPECT_DOUBLE_EQ(u.weight, 1.5);
}

TEST(MCUpdate, ConstructorRejectsNegativeWeight) {
    EXPECT_THROW(update(toy_update {}, "u", -0.5), simplemc_exception);
}

TEST(MCUpdate, ConstructorRejectsEmptyName) {
    EXPECT_THROW(update(toy_update {}, "", 1.0), simplemc_exception);
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

TEST(MCUpdate, GetForwardsToWrapper) {
    toy_update src;
    src.prob = 0.33;
    update u { src, "u", 1.0 };

    auto* p = u.get<toy_update>();
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->prob, 0.33);

    EXPECT_EQ(u.get<doubling_update>(), nullptr);
}

TEST(MCUpdate, SerializationRoundTrip) {
    update u { toy_update {}, "u", 2.0 };
    u.inv_name = "u_inv";
    u.ratio = 0.5;
    u.cumulative_nprops = 11;
    u.cumulative_naccs = 7;
    u.cumulative_nimps = 1;

    mc_serializer s { json_serializer {} };
    auto entry = s["entry"];
    simplemc_save(entry, u);

    update v { toy_update {}, "tmp", 1.0 };
    const auto rentry = mc_serializer { s }["entry"];
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

    mc_serializer s { json_serializer {} };
    auto entry = s["entry"];
    simplemc_save_input_config(entry, u);

    // Loader picks up only "weight"; ratio and cumulative_* are untouched.
    update v { toy_update {}, "u", 1.0 };
    v.ratio = 9.0;
    v.cumulative_nprops = 0;
    const auto rentry = mc_serializer { s }["entry"];
    simplemc_load_input_config(rentry, v);

    EXPECT_DOUBLE_EQ(v.weight, 4.0);
    EXPECT_DOUBLE_EQ(v.ratio, 9.0);          // untouched
    EXPECT_EQ(v.cumulative_nprops, 0u);      // untouched
}
