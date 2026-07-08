#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <limits>
#include <utility>

using namespace simplemc;

// Test default simulation parameters.
TEST(MCSimulationParams, DefaultsAreSensible) {
    simulation_params p;
    EXPECT_EQ(p.steps_per_cycle, 5u);
    EXPECT_EQ(p.cycles_per_check, 1'000'000u);
    EXPECT_EQ(p.max_steps, std::numeric_limits<std::uint64_t>::max());
    EXPECT_DOUBLE_EQ(p.max_time, 10.0);
    EXPECT_NO_THROW(validate_simulation_params(simulation_params {}));
}

// Test that validation throws on invalid simulation parameters.
TEST(MCSimulationParams, ValidateThrowsOnInvalidValues) {
    EXPECT_THROW(validate_simulation_params({ .max_time = -1.0 }), simplemc_exception);
    EXPECT_THROW(validate_simulation_params({ .max_steps = 0 }), simplemc_exception);
    EXPECT_THROW(validate_simulation_params({ .steps_per_cycle = 0 }), simplemc_exception);
    EXPECT_THROW(validate_simulation_params({ .cycles_per_check = 0 }), simplemc_exception);
}

// Test printing of simulation parameters and stats.
TEST(MCSimulationParams, PrintSimulationParams) {
    simulation_params p { .max_steps = 12345, .max_time = 7.5, .steps_per_cycle = 3, .cycles_per_check = 17 };
    print(p);
}

TEST(MCSimulationParams, JsonInputConfigRoundTrip) {
    json_serializer j;

    const simulation_params src { .max_steps = 42,
        .max_time = 1.25,
        .steps_per_cycle = 2,
        .cycles_per_check = 7,
        .skip_measurements = true,
        .checkpoint_after_steps = 100,
        .checkpoint_after_time = 3.5 };
    simplemc_save_input_config(j, src);

    simulation_params dst;
    simplemc_load_input_config(j, dst);

    EXPECT_EQ(dst.max_steps, src.max_steps);
    EXPECT_EQ(dst.max_time, src.max_time);
    EXPECT_EQ(dst.steps_per_cycle, src.steps_per_cycle);
    EXPECT_EQ(dst.cycles_per_check, src.cycles_per_check);
    EXPECT_EQ(dst.skip_measurements, src.skip_measurements);
    EXPECT_EQ(dst.checkpoint_after_steps, src.checkpoint_after_steps);
    EXPECT_EQ(dst.checkpoint_after_time, src.checkpoint_after_time);
}

TEST(MCSimulationParams, JsonInputConfigKeepsDefaultsOnMissingKeys) {
    const json_serializer j { nlohmann::json::object() };

    simulation_params dst;
    EXPECT_NO_THROW(simplemc_load_input_config(j, dst));

    auto def = simulation_params {};
    EXPECT_EQ(dst.steps_per_cycle, def.steps_per_cycle);
    EXPECT_EQ(dst.cycles_per_check, def.cycles_per_check);
    EXPECT_EQ(dst.max_steps, def.max_steps);
    EXPECT_DOUBLE_EQ(dst.max_time, def.max_time);
}

TEST(MCSimulationParams, JsonInputConfigOverridesPartial) {
    nlohmann::json json;
    json["steps_per_cycle"] = 11;
    const json_serializer j { std::move(json) };

    simulation_params dst;
    simplemc_load_input_config(j, dst);

    auto def = simulation_params {};
    EXPECT_EQ(dst.steps_per_cycle, 11u);
    EXPECT_EQ(dst.cycles_per_check, def.cycles_per_check);
    EXPECT_DOUBLE_EQ(dst.max_time, def.max_time);
}

TEST(MCSimulationParams, JsonLoadValidatesInputConfig) {
    nlohmann::json json;
    json["steps_per_cycle"] = 0;
    const json_serializer j { std::move(json) };

    simulation_params dst;
    EXPECT_THROW(simplemc_load_input_config(j, dst), simplemc_exception);
}

// Test default constructed simulation stats.
TEST(MCSimulationStats, DefaultsAreZero) {
    simulation_stats s;
    EXPECT_EQ(s.cumulative_steps, 0u);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 0.0);
}

// Test printing of simulation stats.
TEST(MCSimulationStats, PrintSimulationStats) {
    simulation_stats s { .cumulative_steps = 100, .cumulative_time = 5.25 };
    print(s);
}

// Test folding a simulation context into simulation stats via operator+=.
TEST(MCSimulationStats, AdditionAssignmentFoldsCtx) {
    simulation_stats s { .cumulative_steps = 100, .cumulative_time = 5.25 };

    simulation_ctx ctx;
    ctx.steps_done = 42;
    ctx.runtime = 1.5;
    auto& ref = (s += ctx);

    EXPECT_EQ(&ref, &s);
    EXPECT_EQ(s.cumulative_steps, 142u);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 6.75);
}

// Test that operator+ returns the totals without mutating the stats.
TEST(MCSimulationStats, AdditionReturnsTotalsWithoutMutating) {
    const simulation_stats s { .cumulative_steps = 100, .cumulative_time = 5.25 };

    simulation_ctx ctx;
    ctx.steps_done = 42;
    ctx.runtime = 1.5;
    const auto total = s + ctx;

    EXPECT_EQ(total.cumulative_steps, 142u);
    EXPECT_DOUBLE_EQ(total.cumulative_time, 6.75);
    EXPECT_EQ(s.cumulative_steps, 100u); // untouched
    EXPECT_DOUBLE_EQ(s.cumulative_time, 5.25); // untouched

    // addition is symmetric
    const auto total2 = ctx + s;
    EXPECT_EQ(total2.cumulative_steps, total.cumulative_steps);
    EXPECT_DOUBLE_EQ(total2.cumulative_time, total.cumulative_time);
}

// Test serialization and deserialization of simulation stats.
TEST(MCSimulationStats, JsonStateRoundTrip) {
    json_serializer j;

    const simulation_stats src { .cumulative_steps = 1234, .cumulative_time = 9.75 };
    simplemc_save(j, src);

    simulation_stats dst;
    simplemc_load(j, dst);

    EXPECT_EQ(dst.cumulative_steps, src.cumulative_steps);
    EXPECT_DOUBLE_EQ(dst.cumulative_time, src.cumulative_time);
}

// Test default constructed simulation context.
TEST(MCSimulationCtx, DefaultStepsZeroAndElapsedNonNegative) {
    simulation_ctx x;
    EXPECT_EQ(x.steps_done, 0u);
    EXPECT_EQ(x.runtime, 0.0);
}
