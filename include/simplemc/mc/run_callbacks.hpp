/**
 * @file
 * @brief Callback infrastructure for the simplemc::run loop: no-op defaults and the bundle struct.
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
 * @details `operator()` is a `constexpr` empty function; the compiler inlines it to nothing at
 * `-O1`+, so the run loop emits no extra code when a callback slot is left at its default. Used
 * for the void-returning hooks `on_step`, `on_cycle`, `on_checkpoint`.
 */
struct no_op_callback {
    constexpr void operator()(const simulation_ctx&) const noexcept {}
};

/**
 * @brief Default no-op stop predicate; always returns `false`.
 *
 * @details Used as the default for `stop_when`. When the compiler inlines this, the outer-loop
 * condition `!stop_when(stats)` constant-folds to `true` and drops out of codegen.
 */
struct no_op_stop_callback {
    constexpr bool operator()(const simulation_ctx&) const noexcept { return false; }
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
 * This struct is the canonical model of simplemc::mc_run_callbacks; simplemc::run accepts any
 * type satisfying that concept, so a user-defined bundle with the four hooks works as well.
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

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_RUN_CALLBACKS_HPP
