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
static_assert(mc_update<update<>>);

// A non-conforming type cannot construct an `update`.
static_assert(!std::is_constructible_v<update<>, int>);
static_assert(!std::is_constructible_v<update<>, nothing>);
static_assert(!std::is_constructible_v<update<>, missing_attempt>);
static_assert(!std::is_constructible_v<update<>, missing_accept>);

TEST(MCUpdate, WrapsAndForwardsAttemptAccept) {
    toy_update src;
    src.prob = 0.75;
    auto accepted = src.accepted;

    update u { src };

    EXPECT_DOUBLE_EQ(u.attempt(), 0.75);

    u.accept();
    EXPECT_EQ(*accepted, 1);
}

TEST(MCUpdate, RejectIsNoOpWhenUserTypeOmitsIt) {
    minimal_update src;
    auto accepted = src.accepted;

    update u { src };
    EXPECT_DOUBLE_EQ(u.attempt(), 1.0);

    // should be a no-op: minimal_update has no reject(); the wrapper's reject() must not crash
    // and must not affect any observable state
    u.reject();

    EXPECT_EQ(*accepted, 0);
}

TEST(MCUpdate, RejectForwardsWhenUserTypeProvidesIt) {
    toy_update src;
    auto rejected = src.rejected;

    update u { src };
    u.reject();

    EXPECT_EQ(*rejected, 1);
}

TEST(MCUpdate, CopyProducesIndependentValueSharingCapturedState) {
    toy_update src;
    auto accepted = src.accepted;

    update a { src };
    update b { a }; // deep copy via clone()

    a.accept();
    b.accept();

    // both wrappers' models hold their own copy of toy_update, but each copy retains the same
    // shared_ptr<int>, so the deep copy is visible while the counter is shared
    EXPECT_EQ(*accepted, 2);
}

TEST(MCUpdate, MoveTransfersOwnership) {
    toy_update src;
    auto accepted = src.accepted;

    update a { src };
    update b { std::move(a) };

    b.accept();
    EXPECT_EQ(*accepted, 1);
}

TEST(MCUpdate, CopyAssignReplacesPreviousValue) {
    toy_update first;
    doubling_update second;
    auto first_accepted = first.accepted;
    auto second_committed = second.committed;

    update u { first };
    u = update { second }; // assign a wrapper holding a different concrete type

    u.accept();

    EXPECT_EQ(*first_accepted, 0); // never reached
    EXPECT_EQ(*second_committed, 2); // doubling_update: 1 -> 2
}

TEST(MCUpdate, HeterogeneousStorageInVector) {
    toy_update a;
    doubling_update b;
    auto a_accepted = a.accepted;
    auto b_committed = b.committed;

    std::vector<update<>> v;
    v.emplace_back(a);
    v.emplace_back(b);

    for (auto& u : v) {
        (void)u.attempt();
        u.accept();
    }

    EXPECT_EQ(*a_accepted, 1); // toy_update: 0 -> 1
    EXPECT_EQ(*b_committed, 2); // doubling_update: 1 -> 2
}
