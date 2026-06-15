#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

using namespace simplemc;

namespace {

struct always_accept {
    double attempt() { return 1.0; }
    void accept() {}
    void reject() {}
};

struct counter_meas {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

// Custom kernel: simple counting kernel that never touches an update_set.
// Used to verify the run() driver works with anything satisfying mc_kernel.
struct counting_kernel {
    std::uint64_t calls = 0;
    void step(xoshiro256ss&) { ++calls; }
};

// User-defined callbacks bundle (not run_callbacks) satisfying mc_run_callbacks.
struct counting_callbacks {
    std::shared_ptr<std::uint64_t> steps = std::make_shared<std::uint64_t>(0);
    std::uint64_t stop_after = 0;
    void on_step(const simulation_ctx&) const { ++*steps; }
    void on_cycle(const simulation_ctx&) const {}
    void on_checkpoint(const simulation_ctx&) const {}
    bool stop_when(const simulation_ctx& x) const { return x.steps_done >= stop_after; }
};

static_assert(mc_run_callbacks<run_callbacks<>>);
static_assert(mc_run_callbacks<run_callbacks<progress_printer>>);
static_assert(mc_run_callbacks<counting_callbacks>);
static_assert(!mc_run_callbacks<no_op_callback>);

} // namespace

TEST(MCRun, MaxStepsBounds) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 1 };

    simulation_params p { .max_steps = 1000, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    run(rng, kernel, ms, stats, p);

    // The loop checks bounds only at outer-block boundaries; steps_done may overshoot by up to
    // one block of cycles_per_check * steps_per_cycle = 20 steps.
    EXPECT_GE(stats.last_steps_done, 1000u);
    EXPECT_LE(stats.last_steps_done, 1000u + 20u);
}

TEST(MCRun, MaxTimeBounds) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 2 };

    // Very short time budget; max_steps effectively infinite.
    simulation_params p {
        .max_steps = std::numeric_limits<std::uint64_t>::max(),
        .max_time = 0.05,
        .steps_per_cycle = 1,
        .cycles_per_check = 1,
    };
    run(rng, kernel, ms, stats, p);
    EXPECT_GT(stats.last_steps_done, 0u);
    EXPECT_GE(stats.last_runtime, 0.0);
}

TEST(MCRun, OnStepFiresEverySingleStep) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 3 };

    std::uint64_t step_calls = 0;
    auto cbs = run_callbacks { .on_step = [&](const simulation_ctx&) { ++step_calls; } };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    run(rng, kernel, ms, stats, p, cbs);

    EXPECT_EQ(step_calls, stats.last_steps_done);
}

TEST(MCRun, OnCycleFiresEveryCycle) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 4 };

    std::uint64_t cycle_calls = 0;
    auto cbs = run_callbacks { .on_cycle = [&](const simulation_ctx&) { ++cycle_calls; } };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    run(rng, kernel, ms, stats, p, cbs);

    // Total cycles = steps_done / steps_per_cycle.
    EXPECT_EQ(cycle_calls, stats.last_steps_done / p.steps_per_cycle);
}

TEST(MCRun, StopWhenEndsEarly) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 5 };

    auto cbs = run_callbacks { .stop_when = [](const simulation_ctx& x) {
        return x.steps_done >= 50; // very small budget
    } };

    simulation_params p {
        .max_steps = std::numeric_limits<std::uint64_t>::max(),
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 4,
    };
    run(rng, kernel, ms, stats, p, cbs);
    EXPECT_GE(stats.last_steps_done, 50u);
    EXPECT_LT(stats.last_steps_done, 200u); // stopped early
}

TEST(MCRun, OnCheckpointFiresWhenStepThresholdCrossed) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 6 };

    int checkpoint_calls = 0;
    auto cbs = run_callbacks {
        .on_checkpoint = [&](const simulation_ctx&) { ++checkpoint_calls; },
    };

    // 10 cycles per check * 5 steps = 50 steps per outer-block; checkpoint threshold = 50.
    simulation_params p {
        .max_steps = 200,
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 10,
        .skip_measurements = false,
        .checkpoint_after_steps = 50,
    };
    run(rng, kernel, ms, stats, p, cbs);

    // 200 steps / 50 per checkpoint = 4 (loop exits on max_steps before another fires).
    EXPECT_GE(checkpoint_calls, 1);
    EXPECT_LE(checkpoint_calls, 4);
}

TEST(MCRun, NoCheckpointWhenNoThreshold) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 7 };

    int checkpoint_calls = 0;
    auto cbs = run_callbacks {
        .on_checkpoint = [&](const simulation_ctx&) { ++checkpoint_calls; },
    };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    run(rng, kernel, ms, stats, p, cbs);

    EXPECT_EQ(checkpoint_calls, 0);
}

TEST(MCRun, SkipMeasurementsLeavesCounterUntouched) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    counter_meas src;
    auto count = src.count;
    ms.add({ src, "m", true });
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 8 };

    simulation_params p {
        .max_steps = 100,
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 4,
        .skip_measurements = true,
    };
    run(rng, kernel, ms, stats, p);
    EXPECT_EQ(*count, 0);
}

TEST(MCRun, MeasureAllFiresWhenActive) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    counter_meas src;
    auto count = src.count;
    ms.add({ src, "m", true });
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 9 };

    simulation_params p { .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    run(rng, kernel, ms, stats, p);

    EXPECT_EQ(static_cast<std::uint64_t>(*count), stats.last_steps_done / p.steps_per_cycle);
}

TEST(MCRun, AcceptsUserDefinedCallbacksBundle) {
    update_set us;
    us.add({ always_accept {}, "u", 1.0 });
    measurement_set ms;
    metropolis_kernel kernel { us };
    simulation_stats stats;
    xoshiro256ss rng { 11 };

    counting_callbacks cbs { .stop_after = 50 };

    simulation_params p {
        .max_steps = std::numeric_limits<std::uint64_t>::max(),
        .max_time = 1e6,
        .steps_per_cycle = 5,
        .cycles_per_check = 4,
    };
    run(rng, kernel, ms, stats, p, cbs);

    EXPECT_EQ(*cbs.steps, stats.last_steps_done);
    EXPECT_GE(stats.last_steps_done, 50u);
    EXPECT_LT(stats.last_steps_done, 200u); // stop_when ended the run early
}

TEST(MCRun, AcceptsCustomKernelWithoutUpdateSet) {
    // Custom kernel doesn't touch an update_set; just satisfies the mc_kernel concept.
    counting_kernel kernel;
    measurement_set ms;
    simulation_stats stats;
    xoshiro256ss rng { 10 };

    simulation_params p { .max_steps = 200, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 };
    run(rng, kernel, ms, stats, p);

    EXPECT_GE(stats.last_steps_done, 200u);
    EXPECT_EQ(kernel.calls, stats.last_steps_done);
}
