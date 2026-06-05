#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>

using namespace simplemc;

namespace {

struct always_accept {
    std::shared_ptr<int> attempts = std::make_shared<int>(0);
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() {
        ++*attempts;
        return 1.0;
    }
    void accept() { ++*accepts; }
    void reject() { ++*rejects; }
};

struct half_accept {
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() { return 0.5; }
    void accept() { ++*accepts; }
    void reject() { ++*rejects; }
};

struct impossible_attempt {
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() { return 0.0; }
    void accept() { ++*accepts; }
    void reject() { ++*rejects; }
};

struct counting_measurement {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

} // namespace

TEST(MCSimulation, AddUpdateRejectsDuplicateName) {
    simulation<xoshiro256ss> sim;
    sim.add_update(always_accept {}, "a", 1.0);
    EXPECT_THROW(sim.add_update(always_accept {}, "a", 1.0), simplemc_exception);
}

TEST(MCSimulation, AddUpdateRejectsNegativeWeight) {
    simulation<xoshiro256ss> sim;
    EXPECT_THROW(sim.add_update(always_accept {}, "a", -1.0), simplemc_exception);
}

TEST(MCSimulation, AddUpdatePairSetsInvNameAndRatio) {
    simulation<xoshiro256ss> sim;
    sim.add_update(always_accept {}, "a", 2.0, always_accept {}, "b", 1.0);

    const auto us = sim.update_stats_data();
    ASSERT_EQ(us.size(), 2u);

    EXPECT_EQ(us[0].name, "a");
    EXPECT_EQ(us[0].inv_name, "b");
    EXPECT_DOUBLE_EQ(us[0].weight, 2.0);
    EXPECT_DOUBLE_EQ(us[0].ratio, 0.5);

    EXPECT_EQ(us[1].name, "b");
    EXPECT_EQ(us[1].inv_name, "a");
    EXPECT_DOUBLE_EQ(us[1].weight, 1.0);
    EXPECT_DOUBLE_EQ(us[1].ratio, 2.0);
}

TEST(MCSimulation, AddUpdatePairRejectsAsymmetricZeroWeight) {
    simulation<xoshiro256ss> sim;
    EXPECT_THROW(
        sim.add_update(always_accept {}, "a", 1.0, always_accept {}, "b", 0.0), simplemc_exception);
}

TEST(MCSimulation, AddUpdatePairAllowsBothZero) {
    simulation<xoshiro256ss> sim;
    // both weights zero is allowed; ratios stay at 1.0
    EXPECT_NO_THROW(sim.add_update(always_accept {}, "a", 0.0, always_accept {}, "b", 0.0));
    const auto us = sim.update_stats_data();
    ASSERT_EQ(us.size(), 2u);
    EXPECT_DOUBLE_EQ(us[0].ratio, 1.0);
    EXPECT_DOUBLE_EQ(us[1].ratio, 1.0);
}

TEST(MCSimulation, AddUpdatePairRejectsSameName) {
    simulation<xoshiro256ss> sim;
    EXPECT_THROW(
        sim.add_update(always_accept {}, "a", 1.0, always_accept {}, "a", 1.0), simplemc_exception);
}

TEST(MCSimulation, AddMeasurementRejectsDuplicateName) {
    simulation<xoshiro256ss> sim;
    sim.add_measurement(counting_measurement {}, "m");
    EXPECT_THROW(sim.add_measurement(counting_measurement {}, "m"), simplemc_exception);
}

TEST(MCSimulation, RunThrowsWhenNoUpdates) {
    simulation<xoshiro256ss> sim;
    EXPECT_THROW(
        sim.run({ .max_steps = 10, .max_time = 1.0, .steps_per_cycle = 1, .cycles_per_check = 1 }),
        simplemc_exception);
}

TEST(MCSimulation, RunThrowsWhenAllWeightsZero) {
    simulation<xoshiro256ss> sim;
    sim.add_update(always_accept {}, "a", 0.0);
    EXPECT_THROW(
        sim.run({ .max_steps = 10, .max_time = 1.0, .steps_per_cycle = 1, .cycles_per_check = 1 }),
        simplemc_exception);
}

TEST(MCSimulation, StepIncrementsCountersByOne) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 42 } };
    always_accept aa;
    auto accepts = aa.accepts;
    sim.add_update(aa, "aa", 1.0);
    sim.initialize_update_distribution();

    constexpr int N = 100;
    for (int i = 0; i < N; ++i) {
        sim.step();
    }

    EXPECT_EQ(sim.stats().steps_done, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_stats_data()[0].nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_stats_data()[0].naccs, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_stats_data()[0].nimps, 0u);
    EXPECT_EQ(*accepts, N);
}

TEST(MCSimulation, StepRejectsWhenAttemptBelowUniform) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 7 } };
    half_accept ha;
    auto rejects = ha.rejects;
    sim.add_update(ha, "h", 1.0);
    sim.initialize_update_distribution();

    constexpr int N = 1000;
    for (int i = 0; i < N; ++i) {
        sim.step();
    }

    const auto& us = sim.update_stats_data()[0];
    EXPECT_EQ(us.nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(us.nimps, 0u);
    EXPECT_GT(us.naccs, 0u);
    EXPECT_GT(*rejects, 0);
    EXPECT_EQ(us.naccs + static_cast<std::uint64_t>(*rejects), us.nprops);
}

TEST(MCSimulation, ImpossibleIsRecorded) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    impossible_attempt ia;
    auto accepts = ia.accepts;
    auto rejects = ia.rejects;
    sim.add_update(ia, "imp", 1.0);
    sim.initialize_update_distribution();

    constexpr int N = 50;
    for (int i = 0; i < N; ++i) {
        sim.step();
    }

    const auto& us = sim.update_stats_data()[0];
    EXPECT_EQ(us.nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(us.nimps, static_cast<std::uint64_t>(N));
    EXPECT_EQ(us.naccs, 0u);
    EXPECT_EQ(*accepts, 0);
    EXPECT_EQ(*rejects, 0);
}

TEST(MCSimulation, ZeroWeightUpdateNeverPicked) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 1234 } };
    sim.add_update(always_accept {}, "a", 1.0);
    sim.add_update(always_accept {}, "b", 0.0);
    sim.add_update(always_accept {}, "c", 1.0);
    sim.initialize_update_distribution();

    constexpr int N = 1000;
    for (int i = 0; i < N; ++i) {
        sim.step();
    }

    const auto us = sim.update_stats_data();
    EXPECT_EQ(us[1].nprops, 0u);
    EXPECT_EQ(us[0].nprops + us[2].nprops, static_cast<std::uint64_t>(N));
}

TEST(MCSimulation, CycleCallsMeasureOncePerCycle) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "counter");
    sim.initialize_update_distribution();

    constexpr std::uint64_t steps_per_cycle = 5;
    constexpr int M = 10;
    for (int i = 0; i < M; ++i) {
        sim.cycle(steps_per_cycle);
    }

    EXPECT_EQ(sim.stats().steps_done, static_cast<std::uint64_t>(M * steps_per_cycle));
    EXPECT_EQ(*count, M);
    EXPECT_EQ(sim.update_stats_data()[0].nprops, static_cast<std::uint64_t>(M * steps_per_cycle));
    EXPECT_EQ(sim.measurement_stats_data()[0].name, "counter");
}

TEST(MCSimulation, RunStopsOnMaxSteps) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    constexpr std::uint64_t max_steps = 100;
    constexpr std::uint64_t steps_per_cycle = 3;
    constexpr std::uint64_t cycles_per_check = 5;
    sim.run({ .max_steps = max_steps,
        .max_time = 1000.0,
        .steps_per_cycle = steps_per_cycle,
        .cycles_per_check = cycles_per_check });

    const std::uint64_t block = steps_per_cycle * cycles_per_check;
    EXPECT_GE(sim.stats().steps_done, max_steps);
    EXPECT_LT(sim.stats().steps_done, max_steps + block);
}

TEST(MCSimulation, RunStopsOnMaxTime) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    constexpr double max_time = 0.001;
    constexpr std::uint64_t max_steps = std::numeric_limits<std::uint64_t>::max();
    sim.run({ .max_steps = max_steps, .max_time = max_time, .steps_per_cycle = 50, .cycles_per_check = 200 });

    EXPECT_GE(sim.stats().last_runtime, max_time);
    EXPECT_LT(sim.stats().steps_done, max_steps);
}

TEST(MCSimulation, AccumulateStatsFoldsAllCounters) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    const simulation_params p { .max_steps = 30, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 };
    sim.run(p);

    const std::uint64_t first_steps = sim.stats().steps_done;
    const std::uint64_t first_nprops = sim.update_stats_data()[0].nprops;
    ASSERT_GT(first_steps, 0u);

    sim.accumulate_stats();

    EXPECT_EQ(sim.stats().steps_done, 0u);
    EXPECT_DOUBLE_EQ(sim.stats().last_runtime, 0.0);
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps);
    EXPECT_EQ(sim.update_stats_data()[0].nprops, 0u);
    EXPECT_EQ(sim.update_stats_data()[0].cumulative_nprops, first_nprops);
}

TEST(MCSimulation, RunAutoResetsCurrentCountersAfterAccumulate) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    const simulation_params p { .max_steps = 30, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 };

    sim.run(p);
    const std::uint64_t first_steps = sim.stats().steps_done;
    const std::uint64_t first_nprops = sim.update_stats_data()[0].nprops;

    sim.accumulate_stats();
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps);
    EXPECT_EQ(sim.update_stats_data()[0].cumulative_nprops, first_nprops);

    sim.run(p);
    const std::uint64_t second_steps = sim.stats().steps_done;
    const std::uint64_t second_nprops = sim.update_stats_data()[0].nprops;

    // run() reset current counters at entry — cumulative survived through the second run
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps);
    EXPECT_EQ(sim.update_stats_data()[0].cumulative_nprops, first_nprops);

    sim.accumulate_stats();
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps + second_steps);
    EXPECT_EQ(sim.update_stats_data()[0].cumulative_nprops, first_nprops + second_nprops);
}

TEST(MCSimulation, FindUpdateByName) {
    simulation<xoshiro256ss> sim;
    sim.add_update(always_accept {}, "alpha", 1.0);
    sim.add_update(always_accept {}, "beta", 1.0);
    EXPECT_EQ(sim.find_update("alpha"), 0u);
    EXPECT_EQ(sim.find_update("beta"), 1u);
    EXPECT_THROW((void)sim.find_update("missing"), simplemc_exception);
}

TEST(MCSimulation, FindMeasurementByName) {
    simulation<xoshiro256ss> sim;
    sim.add_measurement(counting_measurement {}, "obs1");
    sim.add_measurement(counting_measurement {}, "obs2");
    EXPECT_EQ(sim.find_measurement("obs1"), 0u);
    EXPECT_EQ(sim.find_measurement("obs2"), 1u);
    EXPECT_THROW((void)sim.find_measurement("missing"), simplemc_exception);
}

TEST(MCSimulation, SetUpdateWeightChangesSelection) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "a", 1.0);
    sim.add_update(always_accept {}, "b", 0.0);
    sim.initialize_update_distribution();

    constexpr int N = 50;
    for (int i = 0; i < N; ++i) {
        sim.step();
    }
    EXPECT_EQ(sim.update_stats_data()[0].nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_stats_data()[1].nprops, 0u);

    sim.set_update_weight("a", 0.0);
    sim.set_update_weight("b", 1.0);
    sim.initialize_update_distribution();

    for (int i = 0; i < N; ++i) {
        sim.step();
    }
    EXPECT_EQ(sim.update_stats_data()[0].nprops, static_cast<std::uint64_t>(N)); // unchanged
    EXPECT_EQ(sim.update_stats_data()[1].nprops, static_cast<std::uint64_t>(N)); // all-second batch
    EXPECT_DOUBLE_EQ(sim.update_stats_data()[0].weight, 0.0);
    EXPECT_DOUBLE_EQ(sim.update_stats_data()[1].weight, 1.0);
}

TEST(MCSimulation, SetUpdateWeightRejectsNegative) {
    simulation<xoshiro256ss> sim;
    sim.add_update(always_accept {}, "a", 1.0);
    EXPECT_THROW(sim.set_update_weight("a", -1.0), simplemc_exception);
    // weight unchanged
    EXPECT_DOUBLE_EQ(sim.update_stats_data()[0].weight, 1.0);
}

TEST(MCSimulation, SetUpdateWeightRejectsMissingName) {
    simulation<xoshiro256ss> sim;
    sim.add_update(always_accept {}, "a", 1.0);
    EXPECT_THROW(sim.set_update_weight("missing", 2.0), simplemc_exception);
}

TEST(MCSimulation, AddMeasurementInactiveSkipsMeasure) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "obs", /*is_active=*/false);
    sim.initialize_update_distribution();

    sim.cycle(5);
    sim.cycle(5);

    EXPECT_EQ(*count, 0);
    EXPECT_FALSE(sim.measurement_stats_data()[0].is_active);
    EXPECT_EQ(sim.stats().steps_done, 10u); // steps still happen
}

TEST(MCSimulation, SetMeasurementActiveTogglesAtRuntime) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "obs");
    sim.initialize_update_distribution();

    sim.cycle(1);
    EXPECT_EQ(*count, 1);

    sim.set_measurement_active("obs", false);
    sim.cycle(1);
    sim.cycle(1);
    EXPECT_EQ(*count, 1); // skipped both times

    sim.set_measurement_active("obs", true);
    sim.cycle(1);
    EXPECT_EQ(*count, 2);
}

TEST(MCSimulation, SetMeasurementActiveRejectsMissingName) {
    simulation<xoshiro256ss> sim;
    sim.add_measurement(counting_measurement {}, "obs");
    EXPECT_THROW(sim.set_measurement_active("missing", false), simplemc_exception);
}

TEST(MCSimulation, ResetStatsZeroesCurrentNotCumulative) {
    simulation<xoshiro256ss> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    sim.run({ .max_steps = 20, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });
    sim.accumulate_stats();
    sim.run({ .max_steps = 20, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });

    const std::uint64_t cumulative_before = sim.update_stats_data()[0].cumulative_nprops;
    const std::uint64_t current_before = sim.update_stats_data()[0].nprops;
    ASSERT_GT(current_before, 0u);
    ASSERT_GT(cumulative_before, 0u);

    sim.reset_stats();

    EXPECT_EQ(sim.update_stats_data()[0].nprops, 0u);
    EXPECT_EQ(sim.update_stats_data()[0].cumulative_nprops, cumulative_before); // survived
    EXPECT_EQ(sim.stats().steps_done, 0u);
}
