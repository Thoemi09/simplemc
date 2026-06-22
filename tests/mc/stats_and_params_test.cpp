#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>

using namespace simplemc;

TEST(MCSimulationParams, DefaultsAreSensible) {
    simulation_params p;
    EXPECT_EQ(p.steps_per_cycle, 5u);
    EXPECT_EQ(p.cycles_per_check, 1'000'000u);
    EXPECT_EQ(p.max_steps, std::numeric_limits<std::uint64_t>::max());
    EXPECT_DOUBLE_EQ(p.max_time, 10.0);
}

TEST(MCSimulationParams, ValidatePassesOnDefault) {
    EXPECT_NO_THROW(validate_simulation_params(simulation_params {}));
}

TEST(MCSimulationParams, ValidateThrowsOnInvalidValues) {
    EXPECT_THROW(validate_simulation_params({ .max_time = -1.0 }), simplemc_exception);
    EXPECT_THROW(validate_simulation_params({ .max_steps = 0 }), simplemc_exception);
    EXPECT_THROW(validate_simulation_params({ .steps_per_cycle = 0 }), simplemc_exception);
    EXPECT_THROW(validate_simulation_params({ .cycles_per_check = 0 }), simplemc_exception);
}

TEST(MCSimulationParams, PrintDoesNotCrash) {
    simulation_params p { .max_steps = 12345, .max_time = 7.5, .steps_per_cycle = 3, .cycles_per_check = 17 };
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    print(f, p);
    std::fclose(f);
}

TEST(MCSimulationStats, DefaultsAreZero) {
    simulation_stats s;
    EXPECT_EQ(s.cumulative_steps, 0u);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 0.0);
}

TEST(MCSimulationCtx, DefaultStepsZeroAndElapsedNonNegative) {
    simulation_ctx x;
    EXPECT_EQ(x.steps_done, 0u);
    x.clk.start();
    EXPECT_GE(x.elapsed(), 0.0);
}

TEST(MCSimulationStats, PrintDoesNotCrash) {
    simulation_stats s { .cumulative_steps = 100, .cumulative_time = 5.25 };
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    print(f, s);
    std::fclose(f);
}

TEST(MCSimulationStats, AccumulateFoldsRunIntoCumulative) {
    simulation_stats s { .cumulative_steps = 100, .cumulative_time = 5.25 };

    simulation_ctx ctx;
    ctx.steps_done = 42;
    ctx.runtime = 1.5;
    accumulate_simulation_stats(s, ctx);

    EXPECT_EQ(s.cumulative_steps, 142u);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 6.75);
}


TEST(MCSimulationParams, JsonStateRoundTrip) {
    const simulation_params src { .max_steps = 123456, .max_time = 7.5, .steps_per_cycle = 3, .cycles_per_check = 17 };

    json_serializer w;
    simplemc_save(w, src);

    simulation_params dst;
    const json_serializer r { w.root() };
    simplemc_load(r, dst);

    EXPECT_EQ(dst.max_steps, simulation_params {}.max_steps);
    EXPECT_DOUBLE_EQ(dst.max_time, simulation_params {}.max_time);
    EXPECT_EQ(dst.steps_per_cycle, src.steps_per_cycle);
    EXPECT_EQ(dst.cycles_per_check, src.cycles_per_check);
}

TEST(MCSimulationParams, JsonInputConfigRoundTrip) {
    const simulation_params src { .max_steps = 42, .max_time = 1.25, .steps_per_cycle = 2, .cycles_per_check = 7 };

    json_serializer w;
    simplemc_save_input_config(w, src);

    simulation_params dst;
    const json_serializer r { w.root() };
    simplemc_load_input_config(r, dst);

    EXPECT_EQ(dst.max_steps, src.max_steps);
    EXPECT_DOUBLE_EQ(dst.max_time, src.max_time);
    EXPECT_EQ(dst.steps_per_cycle, src.steps_per_cycle);
    EXPECT_EQ(dst.cycles_per_check, src.cycles_per_check);
}

TEST(MCSimulationParams, JsonInputConfigKeepsDefaultsOnMissingKeys) {
    // Empty input config — load should leave aggregate defaults untouched and pass validation.
    const json_serializer r { nlohmann::json::object() };

    simulation_params dst;
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst));

    EXPECT_EQ(dst.steps_per_cycle, 5u);
    EXPECT_EQ(dst.cycles_per_check, 1'000'000u);
    EXPECT_EQ(dst.max_steps, std::numeric_limits<std::uint64_t>::max());
    EXPECT_DOUBLE_EQ(dst.max_time, 10.0);
}

TEST(MCSimulationParams, JsonInputConfigOverridesPartial) {
    // Only steps_per_cycle present — others stay at defaults.
    nlohmann::json doc;
    doc["steps_per_cycle"] = 11;
    const json_serializer r { std::move(doc) };

    simulation_params dst;
    simplemc_load_input_config(r, dst);

    EXPECT_EQ(dst.steps_per_cycle, 11u);
    EXPECT_EQ(dst.cycles_per_check, 1'000'000u); // default
    EXPECT_DOUBLE_EQ(dst.max_time, 10.0); // default
}

TEST(MCSimulationParams, JsonLoadValidatesInputConfig) {
    nlohmann::json doc;
    doc["steps_per_cycle"] = 0; // invalid
    const json_serializer r { std::move(doc) };

    simulation_params dst;
    EXPECT_THROW(simplemc_load_input_config(r, dst), simplemc_exception);
}

TEST(MCSimulationStats, JsonRoundTripWritesCumulativeOnly) {
    const simulation_stats src { .cumulative_steps = 1234, .cumulative_time = 9.75 };

    json_serializer w;
    simplemc_save(w, src);

    const auto& root = w.root();
    EXPECT_TRUE(root.contains("cumulative_steps"));
    EXPECT_TRUE(root.contains("cumulative_time"));

    simulation_stats dst;
    const json_serializer r { w.root() };
    simplemc_load(r, dst);

    EXPECT_EQ(dst.cumulative_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst.cumulative_time, 9.75);
}


