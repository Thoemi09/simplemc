#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <gtest/gtest.h>

#include <memory>

using namespace simplemc;

namespace {

struct always_accept {
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    double attempt() { return 1.0; }
    void accept() { ++*accepts; }
    void reject() {}
};

struct always_reject {
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() { return 0.0001; } // > 0 so not impossible, but tiny acceptance prob
    void accept() {}
    void reject() { ++*rejects; }
};

struct always_impossible {
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() { return 0.0; } // impossible
    void accept() {}
    void reject() { ++*rejects; }
};

} // namespace

TEST(MCMetropolisKernel, StepAccepts) {
    update_set us;
    always_accept src;
    auto accepts = src.accepts;
    us.add({ src, "u", 1.0 });

    metropolis_kernel k { us };
    k.prepare();

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 100; ++i) {
        k.step(rng);
    }
    EXPECT_EQ(us[0].nprops, 100u);
    EXPECT_EQ(us[0].naccs, 100u);
    EXPECT_EQ(us[0].nimps, 0u);
    EXPECT_EQ(*accepts, 100);
}

TEST(MCMetropolisKernel, StepImpossible) {
    update_set us;
    always_impossible src;
    auto rejects = src.rejects;
    us.add({ src, "u", 1.0 });

    metropolis_kernel k { us };
    k.prepare();

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 50; ++i) {
        k.step(rng);
    }
    EXPECT_EQ(us[0].nprops, 50u);
    EXPECT_EQ(us[0].naccs, 0u);
    EXPECT_EQ(us[0].nimps, 50u);
    // Impossible proposals are rejected: attempt() is always followed by accept() or reject().
    EXPECT_EQ(*rejects, 50);
}

TEST(MCMetropolisKernel, StepRejects) {
    update_set us;
    always_reject src;
    auto rejects = src.rejects;
    us.add({ src, "u", 1.0 });

    metropolis_kernel k { us };
    k.prepare();

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 200; ++i) {
        k.step(rng);
    }
    EXPECT_EQ(us[0].nprops, 200u);
    // 0.0001 acceptance prob — overwhelmingly likely all 200 reject
    EXPECT_EQ(us[0].nimps, 0u);
    EXPECT_GE(us[0].naccs + *rejects, 200);  // sanity
    EXPECT_LT(us[0].naccs, 5u);              // very few should accept
}

TEST(MCMetropolisKernel, AppliesRatioFromInversePair) {
    // Pair with weights 2 and 4: ratio for first = 4/2 = 2; always_accept returns 1.0, so the
    // effective acceptance probability is min(1, 1.0*2.0)=1.0. The second has ratio 0.5, effective
    // prob = 0.5. Just verify the kernel reads u.ratio (no crash, counters bump).
    update_set us;
    us.add_pair({ always_accept {}, "f", 2.0 }, { always_accept {}, "b", 4.0 });

    metropolis_kernel k { us };
    k.prepare(); // derives the detailed-balance ratios from the current weights
    EXPECT_DOUBLE_EQ(us[0].ratio, 2.0);
    EXPECT_DOUBLE_EQ(us[1].ratio, 0.5);

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 500; ++i) {
        k.step(rng);
    }
    EXPECT_EQ(us[0].nprops + us[1].nprops, 500u);
}

TEST(MCMetropolisKernel, SatisfiesKernelConcept) {
    static_assert(mc_kernel<metropolis_kernel, xoshiro256ss>);
    SUCCEED();
}
