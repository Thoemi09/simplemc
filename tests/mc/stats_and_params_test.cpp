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
    EXPECT_EQ(s.steps_done, 0u);
    EXPECT_EQ(s.cumulative_steps, 0u);
    EXPECT_DOUBLE_EQ(s.last_runtime, 0.0);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 0.0);
}

TEST(MCSimulationStats, PrintDoesNotCrash) {
    simulation_stats s { .steps_done = 42, .last_runtime = 1.5, .cumulative_steps = 100, .cumulative_time = 5.25 };
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    print(f, s);
    std::fclose(f);
}

TEST(MCSimulationStats, ResetZeroesCurrentLeavesCumulative) {
    simulation_stats s { .steps_done = 42, .last_runtime = 1.5, .cumulative_steps = 100, .cumulative_time = 5.25 };

    reset_simulation_stats(s);

    EXPECT_EQ(s.steps_done, 0u);
    EXPECT_DOUBLE_EQ(s.last_runtime, 0.0);
    EXPECT_EQ(s.cumulative_steps, 100u);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 5.25);
}

TEST(MCSimulationStats, AccumulateFoldsAndResets) {
    simulation_stats s { .steps_done = 42, .last_runtime = 1.5, .cumulative_steps = 100, .cumulative_time = 5.25 };

    accumulate_simulation_stats(s);

    EXPECT_EQ(s.steps_done, 0u);
    EXPECT_DOUBLE_EQ(s.last_runtime, 0.0);
    EXPECT_EQ(s.cumulative_steps, 142u);
    EXPECT_DOUBLE_EQ(s.cumulative_time, 6.75);
}

TEST(MCUpdateStats, DefaultsAreSensible) {
    update_stats u;
    EXPECT_EQ(u.name, "");
    EXPECT_EQ(u.inv_name, "");
    EXPECT_DOUBLE_EQ(u.weight, 0.0);
    EXPECT_DOUBLE_EQ(u.ratio, 1.0);
    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.naccs, 0u);
    EXPECT_EQ(u.nimps, 0u);
    EXPECT_EQ(u.cumulative_nprops, 0u);
    EXPECT_EQ(u.cumulative_naccs, 0u);
    EXPECT_EQ(u.cumulative_nimps, 0u);
}

TEST(MCUpdateStats, ResetZeroesCurrentLeavesCumulative) {
    update_stats u { .name = "flip",
        .inv_name = "flip",
        .weight = 2.5,
        .ratio = 1.0,
        .nprops = 100,
        .naccs = 60,
        .nimps = 5,
        .cumulative_nprops = 300,
        .cumulative_naccs = 180,
        .cumulative_nimps = 15 };

    reset_update_stats(u);

    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.naccs, 0u);
    EXPECT_EQ(u.nimps, 0u);
    EXPECT_EQ(u.cumulative_nprops, 300u);
    EXPECT_EQ(u.cumulative_naccs, 180u);
    EXPECT_EQ(u.cumulative_nimps, 15u);

    // configuration is preserved
    EXPECT_EQ(u.name, "flip");
    EXPECT_EQ(u.inv_name, "flip");
    EXPECT_DOUBLE_EQ(u.weight, 2.5);
    EXPECT_DOUBLE_EQ(u.ratio, 1.0);
}

TEST(MCUpdateStats, AccumulateFoldsAndResets) {
    update_stats u { .name = "flip",
        .inv_name = "flip",
        .weight = 2.5,
        .ratio = 1.0,
        .nprops = 100,
        .naccs = 60,
        .nimps = 5,
        .cumulative_nprops = 300,
        .cumulative_naccs = 180,
        .cumulative_nimps = 15 };

    accumulate_update_stats(u);

    EXPECT_EQ(u.nprops, 0u);
    EXPECT_EQ(u.naccs, 0u);
    EXPECT_EQ(u.nimps, 0u);
    EXPECT_EQ(u.cumulative_nprops, 400u);
    EXPECT_EQ(u.cumulative_naccs, 240u);
    EXPECT_EQ(u.cumulative_nimps, 20u);

    // configuration is preserved
    EXPECT_EQ(u.name, "flip");
    EXPECT_EQ(u.inv_name, "flip");
    EXPECT_DOUBLE_EQ(u.weight, 2.5);
    EXPECT_DOUBLE_EQ(u.ratio, 1.0);
}

TEST(MCUpdateStats, PrintDoesNotCrash) {
    const update_stats a {
        .name = "flip", .inv_name = "", .weight = 2.5, .ratio = 1.0, .nprops = 10, .naccs = 6, .nimps = 1
    };
    const update_stats b {
        .name = "shift", .inv_name = "shift_back", .weight = 1.0, .ratio = 0.5, .nprops = 0, .naccs = 0, .nimps = 0
    };

    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const std::array<update_stats, 2> entries { a, b };
    print(f, std::span<const update_stats> { entries });
    std::fclose(f);
}

TEST(MCMeasurementStats, DefaultsAreActiveWithEmptyName) {
    measurement_stats m;
    EXPECT_EQ(m.name, "");
    EXPECT_TRUE(m.is_active);
}

TEST(MCMeasurementStats, PrintDoesNotCrash) {
    const measurement_stats a { .name = "histogram", .is_active = true };
    const measurement_stats b { .name = "legendre", .is_active = false };

    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const std::array<measurement_stats, 2> entries { a, b };
    print(f, std::span<const measurement_stats> { entries });
    std::fclose(f);
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
    const simulation_stats src {
        .steps_done = 7, .last_runtime = 1.5, .cumulative_steps = 1234, .cumulative_time = 9.75
    };

    json_serializer w;
    simplemc_save(w, src);

    // Persistent fields written, current-run fields omitted.
    const auto& root = w.root();
    EXPECT_TRUE(root.contains("cumulative_steps"));
    EXPECT_TRUE(root.contains("cumulative_time"));
    EXPECT_FALSE(root.contains("steps_done"));
    EXPECT_FALSE(root.contains("last_runtime"));

    simulation_stats dst;
    const json_serializer r { w.root() };
    simplemc_load(r, dst);

    EXPECT_EQ(dst.cumulative_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst.cumulative_time, 9.75);
    EXPECT_EQ(dst.steps_done, 0u); // untouched
    EXPECT_DOUBLE_EQ(dst.last_runtime, 0.0);
}

TEST(MCUpdateStats, JsonStateRoundTrip) {
    const update_stats src { .name = "ignored",
        .inv_name = "back",
        .weight = 3.5,
        .ratio = 0.5,
        .nprops = 100,
        .naccs = 60,
        .nimps = 5,
        .cumulative_nprops = 300,
        .cumulative_naccs = 180,
        .cumulative_nimps = 15 };

    json_serializer w;
    simplemc_save(w, src);

    // name and current-run counters are omitted by the hook (parent owns name, current-run resets per run).
    const auto& root = w.root();
    EXPECT_FALSE(root.contains("name"));
    EXPECT_FALSE(root.contains("nprops"));
    EXPECT_FALSE(root.contains("naccs"));
    EXPECT_FALSE(root.contains("nimps"));
    EXPECT_TRUE(root.contains("inv_name"));
    EXPECT_TRUE(root.contains("weight"));
    EXPECT_TRUE(root.contains("ratio"));
    EXPECT_TRUE(root.contains("cumulative_nprops"));

    update_stats dst;
    const json_serializer r { w.root() };
    simplemc_load(r, dst);

    EXPECT_EQ(dst.name, ""); // untouched
    EXPECT_EQ(dst.inv_name, "back");
    EXPECT_DOUBLE_EQ(dst.weight, 3.5);
    EXPECT_DOUBLE_EQ(dst.ratio, 0.5);
    EXPECT_EQ(dst.cumulative_nprops, 300u);
    EXPECT_EQ(dst.cumulative_naccs, 180u);
    EXPECT_EQ(dst.cumulative_nimps, 15u);
    EXPECT_EQ(dst.nprops, 0u); // untouched
}

TEST(MCUpdateStats, JsonInputConfigWritesWeightOnly) {
    const update_stats src { .name = "ignored", .inv_name = "back", .weight = 7.25, .ratio = 0.5 };

    json_serializer w;
    simplemc_save_input_config(w, src);

    const auto& root = w.root();
    EXPECT_TRUE(root.contains("weight"));
    EXPECT_FALSE(root.contains("inv_name"));
    EXPECT_FALSE(root.contains("ratio"));
    EXPECT_FALSE(root.contains("name"));
    EXPECT_FALSE(root.contains("cumulative_nprops"));

    update_stats dst;
    const json_serializer r { w.root() };
    simplemc_load_input_config(r, dst);
    EXPECT_DOUBLE_EQ(dst.weight, 7.25);
}

TEST(MCUpdateStats, JsonInputConfigLoadIsLenient) {
    const json_serializer r { nlohmann::json::object() };

    update_stats dst;
    dst.weight = 4.0;
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst));
    EXPECT_DOUBLE_EQ(dst.weight, 4.0); // unchanged
}

TEST(MCMeasurementStats, JsonStateRoundTrip) {
    const measurement_stats src { .name = "ignored", .is_active = false };

    json_serializer w;
    simplemc_save(w, src);

    const auto& root = w.root();
    EXPECT_TRUE(root.contains("is_active"));
    EXPECT_FALSE(root.contains("name"));

    measurement_stats dst;
    dst.is_active = true;
    const json_serializer r { w.root() };
    simplemc_load(r, dst);

    EXPECT_FALSE(dst.is_active);
    EXPECT_EQ(dst.name, "");
}

TEST(MCMeasurementStats, JsonInputConfigRoundTrip) {
    measurement_stats src;
    src.is_active = false;

    json_serializer w;
    simplemc_save_input_config(w, src);

    measurement_stats dst;
    dst.is_active = true;
    const json_serializer r { w.root() };
    simplemc_load_input_config(r, dst);
    EXPECT_FALSE(dst.is_active);
}

TEST(MCMeasurementStats, JsonInputConfigLoadIsLenient) {
    const json_serializer r { nlohmann::json::object() };
    measurement_stats dst;
    dst.is_active = false;
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst));
    EXPECT_FALSE(dst.is_active); // unchanged
}

