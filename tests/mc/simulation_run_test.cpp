#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>

using namespace simplemc;

TEST(MCSimulationRun, DefaultsHaveRemainingBudget) {
    simulation_run r;

    EXPECT_EQ(r.params().steps_per_cycle, 5u);
    EXPECT_EQ(r.params().cycles_per_check, 1'000'000u);
    EXPECT_EQ(r.params().max_steps, std::numeric_limits<std::uint64_t>::max());
    EXPECT_DOUBLE_EQ(r.params().max_time, 10.0);

    EXPECT_EQ(r.stats().steps_done, 0u);
    EXPECT_EQ(r.stats().cumulative_steps, 0u);
    EXPECT_DOUBLE_EQ(r.stats().last_runtime, 0.0);
    EXPECT_DOUBLE_EQ(r.stats().cumulative_time, 0.0);

    EXPECT_TRUE(r.has_steps_remaining());
    EXPECT_TRUE(r.has_time_remaining());
}

TEST(MCSimulationRun, SetParamsThrowsOnInvalidValues) {
    simulation_run r;

    EXPECT_THROW(r.set_params({ .max_steps = 100, .max_time = -1.0 }), simplemc_exception);
    EXPECT_THROW(r.set_params({ .max_steps = 0 }), simplemc_exception);
    EXPECT_THROW(r.set_params({ .steps_per_cycle = 0 }), simplemc_exception);
    EXPECT_THROW(r.set_params({ .cycles_per_check = 0 }), simplemc_exception);

    // sanity: a valid one does not throw
    EXPECT_NO_THROW(r.set_params({ .max_steps = 100, .max_time = 5.0, .steps_per_cycle = 1, .cycles_per_check = 1 }));
}

TEST(MCSimulationRun, ConstructorThrowsOnInvalidValues) {
    EXPECT_THROW(simulation_run { simulation_params { .max_time = -1.0 } }, simplemc_exception);
    EXPECT_THROW(simulation_run { simulation_params { .max_steps = 0 } }, simplemc_exception);
    EXPECT_THROW(simulation_run { simulation_params { .steps_per_cycle = 0 } }, simplemc_exception);
    EXPECT_THROW(simulation_run { simulation_params { .cycles_per_check = 0 } }, simplemc_exception);
}

TEST(MCSimulationRun, StartStopRecordsLastRuntime) {
    simulation_run r;
    r.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    r.stop();

    EXPECT_GE(r.stats().last_runtime, 0.015); // generous lower bound
    EXPECT_LT(r.stats().last_runtime, 1.0); // generous upper bound
}

TEST(MCSimulationRun, CumulativeTimeAccumulatesAcrossBrackets) {
    simulation_run r;

    r.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    r.stop();
    const double first = r.stats().last_runtime;

    r.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    r.stop();
    const double second = r.stats().last_runtime;

    EXPECT_NEAR(r.stats().cumulative_time, first + second, 1e-9);
}

TEST(MCSimulationRun, HasTimeRemainingFlipsWhenExceeded) {
    simulation_run r { simulation_params { .max_time = 0.01 } };
    r.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_FALSE(r.has_time_remaining());
}

TEST(MCSimulationRun, ResetStatsZerosStatsAndPreservesParams) {
    simulation_run r;
    r.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    r.stop();
    ASSERT_GT(r.stats().last_runtime, 0.0);

    const auto params_before = r.params();
    r.reset_stats();

    EXPECT_EQ(r.stats().steps_done, 0u);
    EXPECT_EQ(r.stats().cumulative_steps, 0u);
    EXPECT_DOUBLE_EQ(r.stats().last_runtime, 0.0);
    EXPECT_DOUBLE_EQ(r.stats().cumulative_time, 0.0);

    EXPECT_EQ(r.params().max_steps, params_before.max_steps);
    EXPECT_DOUBLE_EQ(r.params().max_time, params_before.max_time);
    EXPECT_EQ(r.params().steps_per_cycle, params_before.steps_per_cycle);
    EXPECT_EQ(r.params().cycles_per_check, params_before.cycles_per_check);
}

TEST(MCSimulationRun, JsonRoundTripPreservesParamsAndStats) {
    simulation_run src { simulation_params {
        .max_steps = 12345, .max_time = 7.5, .steps_per_cycle = 3, .cycles_per_check = 17 } };

    // drive some stats so the round trip has something nontrivial to carry
    src.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    src.stop();
    const auto src_stats = src.stats();

    json_serializer writer;
    writer.save_at("run", src);

    json_serializer reader { writer.root() }; // share document via copy
    simulation_run dst;
    reader.load_at("run", dst);

    EXPECT_EQ(dst.params().max_steps, src.params().max_steps);
    EXPECT_DOUBLE_EQ(dst.params().max_time, src.params().max_time);
    EXPECT_EQ(dst.params().steps_per_cycle, src.params().steps_per_cycle);
    EXPECT_EQ(dst.params().cycles_per_check, src.params().cycles_per_check);

    EXPECT_EQ(dst.stats().steps_done, src_stats.steps_done);
    EXPECT_DOUBLE_EQ(dst.stats().last_runtime, src_stats.last_runtime);
    EXPECT_EQ(dst.stats().cumulative_steps, src_stats.cumulative_steps);
    EXPECT_DOUBLE_EQ(dst.stats().cumulative_time, src_stats.cumulative_time);
}
