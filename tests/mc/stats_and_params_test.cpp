#include <simplemc/mc.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>

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
    print(f, a);
    print(f, b);
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
    print(f, a);
    print(f, b);
    const std::array<measurement_stats, 2> entries { a, b };
    print(f, std::span<const measurement_stats> { entries });
    std::fclose(f);
}

