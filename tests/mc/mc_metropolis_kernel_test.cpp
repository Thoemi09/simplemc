// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "./mc_test_utils.hpp"

#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <gtest/gtest.h>

using namespace simplemc;

// Test the Metropolis kernel with an update that is always accepted.
TEST(MCMetropolisKernel, StepAccepts) {
    update_set us { update { dummy_update { .prob = 1.0 }, "u", 1.0 } };
    metropolis_kernel k { us };
    k.prepare();

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 100; ++i) {
        k.step(rng);
    }

    EXPECT_EQ(us.get<0>().stats().nprops, 100u);
    EXPECT_EQ(us.get<0>().stats().naccs, 100u);
    EXPECT_EQ(us.get<0>().stats().nimps, 0u);
    EXPECT_EQ(us.get<0>().value().accepts, 100);
    EXPECT_EQ(us.get<0>().value().rejects, 0);
}

// Test the Metropolis kernel with an update that is always impossible to accept.
TEST(MCMetropolisKernel, StepImpossible) {
    update_set us { update { dummy_update { .prob = -1.0 }, "u", 1.0 } };
    metropolis_kernel k { us };
    k.prepare();

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 50; ++i) {
        k.step(rng);
    }

    EXPECT_EQ(us.get<0>().stats().nprops, 50u);
    EXPECT_EQ(us.get<0>().stats().naccs, 0u);
    EXPECT_EQ(us.get<0>().stats().nimps, 50u);
    EXPECT_EQ(us.get<0>().value().accepts, 0);
    EXPECT_EQ(us.get<0>().value().rejects, 50);
}

// Test the Metropolis kernel with an update that is always rejected.
TEST(MCMetropolisKernel, StepRejects) {
    update_set us { update { dummy_update { .prob = 0.0 }, "u", 1.0 } };
    metropolis_kernel k { us };
    k.prepare();

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 200; ++i) {
        k.step(rng);
    }

    EXPECT_EQ(us.get<0>().stats().nprops, 200u);
    EXPECT_EQ(us.get<0>().stats().nimps, 0u);
    EXPECT_EQ(us.get<0>().stats().naccs, 0u);
    EXPECT_EQ(us.get<0>().value().accepts, 0);
    EXPECT_EQ(us.get<0>().value().rejects, 200);
}

// Test the Metropolis kernel with a pair of updates that have different weights.
TEST(MCMetropolisKernel, AppliesRatioFromInversePair) {
    update_set us { update { dummy_update { .prob = 1.0 }, "f", 2.0 },
        update { dummy_update { .prob = 1.0 }, "b", 4.0 } };
    us.link_pair("f", "b");

    metropolis_kernel k { us };
    k.prepare();
    EXPECT_DOUBLE_EQ(us.get<0>().stats().ratio, 2.0);
    EXPECT_DOUBLE_EQ(us.get<1>().stats().ratio, 0.5);

    xoshiro256ss rng { 42 };
    for (int i = 0; i < 500; ++i) {
        k.step(rng);
    }
    EXPECT_EQ(us.get<0>().stats().nprops + us.get<1>().stats().nprops, 500u);
}

TEST(MCMetropolisKernel, SatisfiesKernelConcept) {
    static_assert(mc_kernel<metropolis_kernel<dummy_update>, xoshiro256ss>);
    SUCCEED();
}
