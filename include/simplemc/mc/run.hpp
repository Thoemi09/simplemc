/**
 * @file
 * @brief Run a Monte Carlo simulation.
 */

#ifndef SIMPLEMC_MC_RUN_HPP
#define SIMPLEMC_MC_RUN_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/run_callbacks.hpp>
#include <simplemc/mc/simulation_ctx.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <cstdint>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief Run a Monte Carlo simulation.
 *
 * @details This function drives every MC simulation. It takes a random number generator, an MC kernel,
 * a set of measurements, simulation parameters and optional callback functions as arguments to do the
 * following:
 *
 * 1. It validates the parameters by calling simplemc::validate_simulation_params().
 * 2. It calls the kernel's `prepare()` method (if it is available).
 * 3. It rebuilds the active set of measurements by calling measurement_set::refresh_active.
 * 4. It initializes a current simplemc::simulation_ctx and other local variables.
 * 5. It performs the actual **MC loop** until some stop criterion is satisfied.
 * 6. It writes the final runtime into the context and returns it.
 *
 * The **MC loop** is a 3-fold nested loop which works as follows:
 *
 * - The innermost loop performs simulation_params::steps_per_cycle MC steps by invoking the kernel's
 * `step(rng)` function, increasing the local simulation_ctx::steps_done counter and calling the
 * `on_step(ctx)` callback function (no-op by default).
 * - The middle loop performs simulation_params::cycles_per_check MC cycles. A single cycle executes
 * the inner loop, calls measurement_set::measure_all() on the given set of measurements and invokes
 * the `on_cycle(ctx)` callback (no-op by default).
 * - The outermost loop refreshes simulation_ctx::runtime and executes the two inner loops and the
 * `on_checkpoint(ctx)` callback (no-op by default). However, in contrast to previous callbacks, this
 * one is only called periodically, either when simulation_params::checkpoint_after_steps MC steps or
 * simulation_params::checkpoint_after_time seconds have passed since the last call. The (outer) loop
 * stops as soon as
 *   - simulation_params::max_steps or more MC steps have been done,
 *   - the run has taken simulation_params::max_time seconds or longer,
 *   - the user-defined `stop_when()` callback returns true.
 *
 * On exit, the returned simplemc::simulation_ctx holds the final simulation_ctx::steps_done and
 * simulation_ctx::runtime of the completed run. Per-update counters are written by the kernel and the
 * update wrapper.
 *
 * @note Successful simulations always perform a number of MC steps which is a multiple of
 * simulation_params::cycles_per_check times simulation_params::steps_per_cycle.
 *
 * @tparam RNG Random number generator type.
 * @tparam Kernel simplemc::mc_kernel type.
 * @tparam Ms User measurement types held by the measurement set (deduced).
 * @tparam Cbs simplemc::mc_run_callbacks bundle type.
 * @param rng Random number generator.
 * @param kernel Kernel that implements the desired MC algorithm.
 * @param meas Measurement set. Its active cache is rebuilt at entry.
 * @param p Simulation parameters controlling the run.
 * @param cbs Optional callbacks.
 * @return The final simplemc::simulation_ctx of the run containing the total MC steps done and the
 * total runtime.
 */
template <typename RNG, typename Kernel, mc_measurement... Ms, typename Cbs = run_callbacks<>>
    requires mc_kernel<std::remove_cvref_t<Kernel>, RNG> && mc_run_callbacks<std::remove_cvref_t<Cbs>>
[[nodiscard]] simulation_ctx run(
    RNG& rng, Kernel&& kernel, measurement_set<Ms...>& meas, const simulation_params& p, Cbs&& cbs = {}) { // NOLINT
    // validate parameters
    validate_simulation_params(p);

    // optionally prepare the kernel
    if constexpr (requires { kernel.prepare(); }) {
        kernel.prepare();
    }

    // reset active measurements
    if (p.skip_measurements) {
        meas.clear_active();
    } else {
        meas.refresh_active();
    }

    // simulation context + local variables
    simulation_ctx ctx;
    ctx.clk.start();
    std::uint64_t steps_since_ck = 0;
    double time_at_last_ck = 0.0;

    // MC loop
    while (ctx.steps_done < p.max_steps && ctx.runtime < p.max_time && !cbs.stop_when(ctx)) {
        for (std::uint64_t c = 0; c < p.cycles_per_check; ++c) {
            for (std::uint64_t s = 0; s < p.steps_per_cycle; ++s) {
                kernel.step(rng);
                ++ctx.steps_done;
                cbs.on_step(ctx);
            }
            meas.measure_all();
            cbs.on_cycle(ctx);
        }
        ctx.runtime = ctx.elapsed();

        // checkpoint callback
        steps_since_ck += p.steps_per_cycle * p.cycles_per_check;
        if (steps_since_ck >= p.checkpoint_after_steps || ctx.runtime - time_at_last_ck >= p.checkpoint_after_time) {
            cbs.on_checkpoint(ctx);
            steps_since_ck = 0;
            time_at_last_ck = ctx.runtime;
        }
    }

    // write the final runtime and hand back the context
    ctx.runtime = ctx.elapsed();
    return ctx;
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_RUN_HPP
