#include "./mc_test_utils.hpp"

#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json.hpp>
#include <simplemc/utils/file_io.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <tuple>
#include <utility>

using namespace simplemc;

namespace {

// Custom kernel: simple counting kernel that never touches an update_set.
// Used to verify the run() driver works with anything satisfying mc_kernel.
struct counting_kernel {
    std::uint64_t calls = 0;
    void step(xoshiro256ss&) { ++calls; }
};

// Stateful update with an ADL simplemc_save / simplemc_load, for the in-run checkpoint round-trip.
struct stateful_update {
    int counter = 0;
    double attempt() {
        ++counter;
        return 1.0;
    }
    void accept() {}
};

template <serializer S>
void simplemc_save(S& s, const stateful_update& u) {
    s.save_at("counter", u.counter);
}

template <serializer S>
void simplemc_load(const S& s, stateful_update& u) {
    s.load_at("counter", u.counter);
}

// User-defined callbacks bundle (not run_callbacks) satisfying mc_run_callbacks. The concept only
// requires callability on a non-const bundle, so on_step can mutate a plain member.
struct counting_callbacks {
    std::uint64_t steps = 0;
    std::uint64_t stop_after = 0;
    void on_step(const simulation_ctx&) { ++steps; }
    void on_cycle(const simulation_ctx&) const {}
    void on_checkpoint(const simulation_ctx&) const {}
    [[nodiscard]] bool stop_when(const simulation_ctx& x) const { return x.steps_done >= stop_after; }
};

static_assert(mc_run_callbacks<run_callbacks<>>);
static_assert(mc_run_callbacks<run_callbacks<progress_printer>>);
static_assert(mc_run_callbacks<counting_callbacks>);
static_assert(!mc_run_callbacks<no_op_callback>);

} // namespace

// Test that the run driver stops after max. steps.
TEST(MCRun, MaxStepsBounds) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 1 };

    simulation_params p { .max_steps = 1000, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    const auto ctx = run(rng, kernel, ms, p);

    EXPECT_EQ(ctx.steps_done, 1000u);
}

// Test that the run driver stops after max. time.
TEST(MCRun, MaxTimeBounds) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 2 };

    simulation_params p {
        .max_steps = std::numeric_limits<std::uint64_t>::max(),
        .max_time = 0.05,
        .steps_per_cycle = 1,
        .cycles_per_check = 1,
    };
    const auto ctx = run(rng, kernel, ms, p);

    EXPECT_GT(ctx.steps_done, 0u);
    EXPECT_GE(ctx.runtime, 0.05);
}

// Test that the on_step callback is called after every single step.
TEST(MCRun, OnStepFiresEverySingleStep) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 3 };

    std::uint64_t step_calls = 0;
    auto cbs = run_callbacks { .on_step = [&](const simulation_ctx&) { ++step_calls; } };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    const auto ctx = run(rng, kernel, ms, p, cbs);

    EXPECT_EQ(step_calls, ctx.steps_done);
}

// Test that the on_cycle callback is called after every cycle.
TEST(MCRun, OnCycleFiresEveryCycle) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 4 };

    std::uint64_t cycle_calls = 0;
    auto cbs = run_callbacks { .on_cycle = [&](const simulation_ctx&) { ++cycle_calls; } };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    const auto ctx = run(rng, kernel, ms, p, cbs);

    EXPECT_EQ(cycle_calls, ctx.steps_done / p.steps_per_cycle);
}

// Test that the stop_when callback can terminate the run early.
TEST(MCRun, StopWhenEndsEarly) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 5 };

    auto cbs = run_callbacks { .stop_when = [](const simulation_ctx& x) { return x.steps_done >= 50; } };

    simulation_params p {
        .max_steps = 200,
        .max_time = 1e6,
        .steps_per_cycle = 1,
        .cycles_per_check = 1,
    };
    const auto ctx = run(rng, kernel, ms, p, cbs);

    EXPECT_EQ(ctx.steps_done, 50u);
}

// Test that the on_checkpoint callback is called when the step threshold is crossed.
TEST(MCRun, OnCheckpointFiresWhenStepThresholdCrossed) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 6 };

    int checkpoint_calls = 0;
    auto cbs = run_callbacks {
        .on_checkpoint = [&](const simulation_ctx&) { ++checkpoint_calls; },
    };

    // 10 cycles per check * 5 steps = 50 steps per outer-block; checkpoint threshold = 50
    simulation_params p {
        .max_steps = 200,
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 10,
        .skip_measurements = false,
        .checkpoint_after_steps = 50,
    };
    std::ignore = run(rng, kernel, ms, p, cbs);

    // 200 steps / 50 per checkpoint = 4 (loop exits on max_steps before another fires)
    EXPECT_EQ(checkpoint_calls, 4);
}

// Test that the on_checkpoint callback is not called when no threshold is set.
TEST(MCRun, NoCheckpointWhenNoThreshold) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 7 };

    int checkpoint_calls = 0;
    auto cbs = run_callbacks {
        .on_checkpoint = [&](const simulation_ctx&) { ++checkpoint_calls; },
    };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    std::ignore = run(rng, kernel, ms, p, cbs);

    EXPECT_EQ(checkpoint_calls, 0);
}

// Test that if skip_measurements=true, nothing is being measured.
TEST(MCRun, SkipMeasurementsLeavesCounterUntouched) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set ms { measurement { dummy_measurement {}, "m", true } };
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 8 };

    simulation_params p {
        .max_steps = 100,
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 4,
        .skip_measurements = true,
    };
    std::ignore = run(rng, kernel, ms, p);

    EXPECT_EQ(ms.get<0>().value().count, 0);
}

// Test that if skip_measurements=false, measurements are taken after every cycle.
TEST(MCRun, MeasureAllFiresWhenActive) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set ms { measurement { dummy_measurement {}, "m", true } };
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 9 };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    const auto ctx = run(rng, kernel, ms, p);

    EXPECT_EQ(ms.get<0>().value().count, ctx.steps_done / p.steps_per_cycle);
}

// Test run() with a user-defined callbacks bundle that satisfies mc_run_callbacks.
TEST(MCRun, AcceptsUserDefinedCallbacksBundle) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 11 };

    counting_callbacks cbs { .stop_after = 50 };

    simulation_params p {
        .max_steps = 200,
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 4,
    };
    const auto ctx = run(rng, kernel, ms, p, cbs);

    EXPECT_EQ(cbs.steps, ctx.steps_done);
    EXPECT_EQ(ctx.steps_done, 60u);
}

// Test run() with a custom kernel.
TEST(MCRun, AcceptsCustomKernelWithoutUpdateSet) {
    counting_kernel kernel;
    measurement_set<> ms;
    xoshiro256ss rng { 10 };

    simulation_params p { .max_steps = 200, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    const auto ctx = run(rng, kernel, ms, p);

    EXPECT_EQ(ctx.steps_done, 200u);
    EXPECT_EQ(kernel.calls, ctx.steps_done);
}

// Test that a checkpoint-writing on_checkpoint lambda can be used inside run() to write a file, and
// that the file can be loaded into fresh components to restore the state.
TEST(MCRun, CheckpointWriterInsideRunRoundTrips) {
    update_set us { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 21 };
    simulation_stats stats;

    const auto path = std::filesystem::temp_directory_path() / "mc_run_writer_roundtrip.json";
    std::filesystem::remove(path);
    auto cbs = run_callbacks { .on_checkpoint = [&](const simulation_ctx&) {
        atomic_file_write(path, [&](const std::filesystem::path& tmp) {
            json_serializer s;
            simplemc_save(s, rng, us, ms, stats);
            write_json_file(s.root(), tmp);
        });
    } };

    simulation_params p {
        .max_steps = 100,
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 5,
        .checkpoint_after_steps = 25,
    };
    std::ignore = run(rng, kernel, ms, p, cbs);
    ASSERT_TRUE(std::filesystem::exists(path));

    xoshiro256ss dst_rng { 999 };
    update_set dst_us { update { stateful_update {}, "u", 2.0 } };
    measurement_set<> dst_ms;
    simulation_stats dst_stats;

    json_serializer reader;
    read_json_file(reader.root(), path);
    simplemc_load(reader, dst_rng, dst_us, dst_ms, dst_stats);

    const int dst_counter = dst_us.get<0>().value().counter;
    EXPECT_DOUBLE_EQ(dst_us.get<0>().stats().weight, 1.0);
    EXPECT_EQ(dst_counter, 100);
    EXPECT_EQ(dst_counter, us.get<0>().value().counter);
    std::filesystem::remove(path);
}

// Test the progress printer as a callback inside run().
TEST(MCRun, ProgressPrinterRunsAsCallback) {
    update_set us { update { dummy_update {}, "u", 1.0 } };
    measurement_set<> ms;
    metropolis_kernel kernel { us };
    xoshiro256ss rng { 22 };

    simulation_params p { .max_steps = 100, .max_time = 1, .steps_per_cycle = 5, .cycles_per_check = 4 };
    auto pl = progress_line { .max_steps = p.max_steps, .max_time = p.max_time, .show_bar = true };
    auto cbs = run_callbacks { .on_cycle = progress_printer { pl, false } };
    const auto ctx = run(rng, kernel, ms, p, cbs);
    EXPECT_EQ(ctx.steps_done, 100u);
}
