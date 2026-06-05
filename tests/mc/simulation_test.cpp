#include <simplemc/mc.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <limits>

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

TEST(MCSimulationParams, JsonRoundTripPreservesAllFields) {
    const simulation_params src { .max_steps = 12345, .max_time = 7.5, .steps_per_cycle = 3, .cycles_per_check = 17 };

    json_serializer writer;
    writer.save_at("params", src);

    json_serializer reader { writer.root() };
    simulation_params dst;
    reader.load_at("params", dst);

    EXPECT_EQ(dst.max_steps, src.max_steps);
    EXPECT_DOUBLE_EQ(dst.max_time, src.max_time);
    EXPECT_EQ(dst.steps_per_cycle, src.steps_per_cycle);
    EXPECT_EQ(dst.cycles_per_check, src.cycles_per_check);
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

TEST(MCSimulationStats, JsonRoundTripPreservesAllFields) {
    const simulation_stats src {
        .steps_done = 42, .last_runtime = 1.5, .cumulative_steps = 100, .cumulative_time = 5.25
    };

    json_serializer writer;
    writer.save_at("stats", src);

    json_serializer reader { writer.root() };
    simulation_stats dst;
    reader.load_at("stats", dst);

    EXPECT_EQ(dst.steps_done, src.steps_done);
    EXPECT_DOUBLE_EQ(dst.last_runtime, src.last_runtime);
    EXPECT_EQ(dst.cumulative_steps, src.cumulative_steps);
    EXPECT_DOUBLE_EQ(dst.cumulative_time, src.cumulative_time);
}
