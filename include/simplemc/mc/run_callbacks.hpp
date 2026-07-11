// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Default callbacks for simplemc::run.
 */

#ifndef SIMPLEMC_MC_RUN_CALLBACKS_HPP
#define SIMPLEMC_MC_RUN_CALLBACKS_HPP

#include <simplemc/mc/simulation_ctx.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-callbacks
 * @{
 */

/**
 * @brief Default no-op callable used when the user does not provide a callback.
 *
 * @details It can be used for the void-returning hooks run_callbacks::on_step,
 * run_callbacks::on_cycle and run_callbacks::on_checkpoint.
 */
struct no_op_callback {
    /**
     * @brief No-op that returns void.
     */
    constexpr void operator()(const simulation_ctx&) const noexcept {}
};

/**
 * @brief Default no-op stop predicate always returning `false`.
 *
 * @details It can be used for the run_callbacks::stop_when hook.
 */
struct no_op_stop_callback {
    /**
     * @brief No-op that always returns `false`.
     */
    constexpr bool operator()(const simulation_ctx&) const noexcept { return false; }
};

/**
 * @brief Bundle of optional callbacks accepted by simplemc::run.
 *
 * @details It satisfies simplemc::mc_run_callbacks. Each field is a callable. Default-constructed
 * instances are no-ops.
 *
 * Users can construct a callback bundle with designated initializers and lambdas:
 *
 * @code{.cpp}
 * auto cbs = simplemc::run_callbacks{ .on_cycle = [&](const auto& s) { print(s); } };
 * @endcode
 *
 * @tparam StepCB Type of the per-step callback.
 * @tparam CycleCB Type of the per-cycle callback.
 * @tparam CheckpointCB Type of the per-checkpoint callback.
 * @tparam StopCB Type of the early-stop predicate.
 */
template <typename StepCB = no_op_callback, typename CycleCB = no_op_callback, typename CheckpointCB = no_op_callback,
    typename StopCB = no_op_stop_callback>
struct run_callbacks {
    /**
     * @brief Callback invoked after every MC kernel step.
     */
    [[no_unique_address]] StepCB on_step {};

    /**
     * @brief Callback invoked after every MC cycle.
     */
    [[no_unique_address]] CycleCB on_cycle {};

    /**
     * @brief Callback invoked periodically depending on simulation_params::checkpoint_after_steps and
     * simulation_params::checkpoint_after_time.
     */
    [[no_unique_address]] CheckpointCB on_checkpoint {};

    /**
     * @brief Callback invoked in the outer `while` condition returning either `false` to keep the MC
     * simulation going or `true` to stop the simulation.
     */
    [[no_unique_address]] StopCB stop_when {};
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_RUN_CALLBACKS_HPP
