/**
 * @file
 * @brief Free run-loop driver and the callbacks struct it accepts.
 */

#ifndef SIMPLEMC_MC_RUN_HPP
#define SIMPLEMC_MC_RUN_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <cstdint>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Default no-op callable used when the user does not provide a callback.
 *
 * @details `operator()` is a `constexpr` empty function; the compiler inlines it to nothing at
 * `-O1`+, so the run loop emits no extra code when a callback slot is left at its default. Used
 * for the void-returning hooks `on_step`, `on_cycle`, `on_checkpoint`.
 */
struct no_op_callback {
    constexpr void operator()(const simulation_stats&) const noexcept {}
};

/**
 * @brief Default no-op stop predicate; always returns `false`.
 *
 * @details Used as the default for `stop_when`. When the compiler inlines this, the outer-loop
 * condition `!stop_when(stats)` constant-folds to `true` and drops out of codegen.
 */
struct no_op_stop_callback {
    constexpr bool operator()(const simulation_stats&) const noexcept { return false; }
};

/**
 * @brief Bundle of optional callbacks accepted by simplemc::run.
 *
 * @details Each field is a callable; default-constructed instances are no-ops, so leaving slots
 * unset has zero runtime cost at `-O1`+. C++20 aggregate-CTAD with designated initializers lets
 * the user write any subset:
 *
 * @code{.cpp}
 * auto cbs = simplemc::run_callbacks{ .on_cycle = [&](const auto& s) { print(s); } };
 * @endcode
 *
 * Firing cadence:
 *   - `on_step`       — after every kernel.step(rng), inside the inner step loop.
 *   - `on_cycle`      — after each cycle's measurement sweep (cycle-boundary).
 *   - `on_checkpoint` — at most once per cycles_per_check, gated by
 *                       `params.checkpoint_after_steps` and `params.checkpoint_after_time`.
 *   - `stop_when`     — once per cycles_per_check, in the outer while condition; returns true to
 *                       stop early.
 *
 * @tparam StepCb       Type of the per-step callback. Defaults to simplemc::no_op_callback.
 * @tparam CycleCb      Type of the per-cycle callback. Defaults to simplemc::no_op_callback.
 * @tparam CheckpointCb Type of the per-checkpoint callback. Defaults to simplemc::no_op_callback.
 * @tparam StopCb       Type of the early-stop predicate. Defaults to simplemc::no_op_stop_callback.
 */
template <class StepCb = no_op_callback, class CycleCb = no_op_callback,
    class CheckpointCb = no_op_callback, class StopCb = no_op_stop_callback>
struct run_callbacks {
    [[no_unique_address]] StepCb on_step {};
    [[no_unique_address]] CycleCb on_cycle {};
    [[no_unique_address]] CheckpointCb on_checkpoint {};
    [[no_unique_address]] StopCb stop_when {};
};

/**
 * @brief Free run-loop driver that advances a kernel over a measurement set.
 *
 * @details Drives the standard nested loop: outer `while` polls the stop criteria
 * (`stats.steps_done < params.max_steps`, `stats.last_runtime < params.max_time`, and
 * `!cbs.stop_when(stats)`); inner `for (cycles_per_check)` runs that many cycles; each cycle runs
 * `params.steps_per_cycle` Metropolis steps via `kernel.step(rng)` and then invokes
 * `meas.measure_all()` (unless `params.skip_measurements`).
 *
 * Before the loop, validates `params` and (if the kernel exposes one) calls `kernel.prepare()`.
 * Resets the per-run counters on `stats`. Builds the active-set cache on `meas` from each entry's
 * `is_active` flag (cleared instead if `skip_measurements`).
 *
 * After each block of `cycles_per_check` cycles, `stats.last_runtime` is refreshed from the
 * wall-clock timer, the checkpoint thresholds are checked against the elapsed step / time deltas
 * since the last checkpoint, and `on_checkpoint` is invoked if either threshold is crossed.
 *
 * `stats.steps_done` is bumped here on every step; per-update counters (`nprops`, `naccs`,
 * `nimps`) are written by the kernel.
 *
 * @tparam Kernel       Step kernel satisfying simplemc::mc_kernel for the given RNG.
 * @tparam S1, S2       Serializer types carried by the measurement_set.
 * @tparam RNG          Random number generator type.
 * @tparam StepCb, CycleCb, CheckpointCb, StopCb  Callback types (see simplemc::run_callbacks).
 *
 * @param kernel Kernel that performs each step.
 * @param meas Measurement set; its active cache is rebuilt at entry.
 * @param p Parameters controlling the run.
 * @param stats Aggregate simulation statistics; reset at entry.
 * @param rng RNG threaded into both the kernel and the measurement sweep.
 * @param cbs Optional callbacks; default = all no-ops.
 */
template <class Kernel, serializer S1, serializer S2, class RNG, class StepCb = no_op_callback,
    class CycleCb = no_op_callback, class CheckpointCb = no_op_callback,
    class StopCb = no_op_stop_callback>
    requires mc_kernel<Kernel, RNG>
void run(Kernel& kernel, measurement_set<S1, S2>& meas, const simulation_params& p,
    simulation_stats& stats, RNG& rng,
    const run_callbacks<StepCb, CycleCb, CheckpointCb, StopCb>& cbs = {}) {
    validate_simulation_params(p);
    if constexpr (requires { kernel.prepare(); }) {
        kernel.prepare();
    }
    reset_simulation_stats(stats);
    if (p.skip_measurements) {
        meas.clear_active();
    } else {
        meas.refresh_active();
    }

    timer<> clk;
    clk.start();
    std::uint64_t steps_since_ck = 0;
    double time_at_last_ck = 0.0;

    while (stats.steps_done < p.max_steps && stats.last_runtime < p.max_time
        && !cbs.stop_when(stats)) {
        for (std::uint64_t c = 0; c < p.cycles_per_check; ++c) {
            for (std::uint64_t s = 0; s < p.steps_per_cycle; ++s) {
                kernel.step(rng);
                ++stats.steps_done;
                cbs.on_step(stats);
            }
            if (!p.skip_measurements) {
                meas.measure_all();
            }
            cbs.on_cycle(stats);
        }
        stats.last_runtime = clk.elapsed();

        steps_since_ck += p.steps_per_cycle * p.cycles_per_check;
        if (steps_since_ck >= p.checkpoint_after_steps
            || stats.last_runtime - time_at_last_ck >= p.checkpoint_after_time) {
            cbs.on_checkpoint(stats);
            steps_since_ck = 0;
            time_at_last_ck = stats.last_runtime;
        }
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_RUN_HPP
