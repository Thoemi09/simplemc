/**
 * @file
 * @brief Free run-loop driver that advances a kernel over a measurement set.
 */

#ifndef SIMPLEMC_MC_RUN_HPP
#define SIMPLEMC_MC_RUN_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/run_callbacks.hpp>
#include <simplemc/mc/simulation_ctx.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <cstdint>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief Free run-loop driver that advances a kernel over a measurement set.
 *
 * @details Creates a fresh simplemc::simulation_ctx for this run (live step counter + wall-clock)
 * and drives the standard nested loop: outer `while` polls the stop criteria
 * (`ctx.steps_done < params.max_steps`, `ctx.elapsed() < params.max_time`, and
 * `!cbs.stop_when(ctx)`); inner `for (cycles_per_check)` runs that many cycles; each cycle runs
 * `params.steps_per_cycle` Metropolis steps via `kernel.step(rng)` and then invokes
 * `meas.measure_all()` (a no-op when `params.skip_measurements` cleared the active cache at
 * entry). All four callbacks receive `ctx`, so a hook can read the live step count
 * (`ctx.steps_done`) and live wall-clock (`ctx.elapsed()`).
 *
 * Before the loop, validates `params` and (if the kernel exposes one) calls `kernel.prepare()`.
 * Builds the active-set cache on `meas` from each entry's `is_active` flag (cleared instead if
 * `skip_measurements`).
 *
 * After each block of `cycles_per_check` cycles, the checkpoint thresholds are checked against the
 * elapsed step / time deltas since the last checkpoint, and `on_checkpoint` is invoked if either
 * threshold is crossed.
 *
 * @note `max_time` and `stop_when` are polled only once per `cycles_per_check` block, i.e. every
 * `steps_per_cycle * cycles_per_check` steps. With the default `cycles_per_check` of \f$ 10^6 \f$,
 * a slow step function can overshoot `max_time` considerably — tune `cycles_per_check` to the cost
 * of a step.
 *
 * On exit, the run's final totals are recorded into `stats`: `last_steps_done = ctx.steps_done`
 * and `last_runtime = ctx.elapsed()`. The cumulative fields on `stats` are left untouched (fold
 * them in with simplemc::accumulate_simulation_stats). Per-update counters (`nprops`, `naccs`,
 * `nimps`) are written by the kernel; note the asymmetry with `stats.last_*`: the per-update
 * current-run counters are **not** reset at run entry — they keep accumulating until
 * `accumulate_counters()` / `reset_run_counters()` is called on the update set.
 *
 * @tparam Kernel Step kernel satisfying simplemc::mc_kernel for the given RNG.
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like, carried by the measurement_set.
 * @tparam RNG    Random number generator type.
 * @tparam Cbs    Callbacks bundle satisfying simplemc::mc_run_callbacks. Defaults to
 *                simplemc::run_callbacks<> (all no-ops); any user type with the four hooks works.
 *
 * @param kernel Kernel that performs each step.
 * @param meas Measurement set; its active cache is rebuilt at entry.
 * @param p Parameters controlling the run.
 * @param stats Recorded statistics; the run's final totals are written to `last_*` at exit.
 * @param rng RNG threaded into the kernel's `step()`.
 * @param cbs Optional callbacks; default = all no-ops.
 */
template <class Kernel, mc_traits_like Traits, class RNG,
    mc_run_callbacks Cbs = run_callbacks<>>
    requires mc_kernel<Kernel, RNG>
void run(Kernel& kernel, measurement_set<Traits>& meas, const simulation_params& p,
    simulation_stats& stats, RNG& rng, const Cbs& cbs = {}) {
    validate_simulation_params(p);
    if constexpr (requires { kernel.prepare(); }) {
        kernel.prepare();
    }
    if (p.skip_measurements) {
        meas.clear_active();
    } else {
        meas.refresh_active();
    }

    simulation_ctx ctx;
    ctx.clk.start();
    std::uint64_t steps_since_ck = 0;
    double time_at_last_ck = 0.0;
    double last_runtime = 0.0;

    while (ctx.steps_done < p.max_steps && last_runtime < p.max_time && !cbs.stop_when(ctx)) {
        for (std::uint64_t c = 0; c < p.cycles_per_check; ++c) {
            for (std::uint64_t s = 0; s < p.steps_per_cycle; ++s) {
                kernel.step(rng);
                ++ctx.steps_done;
                cbs.on_step(ctx);
            }
            meas.measure_all();
            cbs.on_cycle(ctx);
        }
        last_runtime = ctx.elapsed();

        steps_since_ck += p.steps_per_cycle * p.cycles_per_check;
        if (steps_since_ck >= p.checkpoint_after_steps
            || last_runtime - time_at_last_ck >= p.checkpoint_after_time) {
            cbs.on_checkpoint(ctx);
            steps_since_ck = 0;
            time_at_last_ck = last_runtime;
        }
    }

    stats.last_steps_done = ctx.steps_done;
    stats.last_runtime = ctx.elapsed();
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_RUN_HPP
