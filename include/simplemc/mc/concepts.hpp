/**
 * @file
 * @brief Concepts that describe the user-facing contracts of simplemc-mc.
 */

#ifndef SIMPLEMC_MC_CONCEPTS_HPP
#define SIMPLEMC_MC_CONCEPTS_HPP

#include <simplemc/mc/simulation_ctx.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>

#include <concepts>

namespace simplemc {

/**
 * @ingroup simplemc-mc-entities-measurements
 * @brief Contract a type must satisfy to be wrapped in a simplemc::measurement.
 *
 * @details A Monte Carlo measurement is an observer of the simulation state that simplemc::run
 * invokes at the end of every MC cycle.
 *
 * Let `m` be an instance of type `M`. The requirements for a type `M` to be a measurement are the
 * following:
 *
 * - `m.measure()` performs the measurement.
 *
 * @tparam M Type to check.
 */
template <typename M>
concept mc_measurement = requires(M& m) { m.measure(); };

/**
 * @ingroup simplemc-mc-entities-updates
 * @brief Contract a type must satisfy to be wrapped in a simplemc::update.
 *
 * @details A Monte Carlo update proposes a change to the simulation state via `attempt()`, which
 * returns the acceptance probability of the proposed change. The driver then either commits the
 * change by calling `accept()` or rolls it back by calling `reject()` (the latter is optional and
 * falls back to a no-op).
 *
 * Let `u` be an instance of type `U`. The requirements for a type `U` to be an update are the
 * following:
 *
 * - `u.attempt()` proposes a change to the simulation state and returns the acceptance probability
 * as a type convertible to `double`.
 * - `u.accept()` commits the proposed change to the simulation state.
 *
 * @tparam U Type to check.
 */
template <typename U>
concept mc_update = requires(U& u) {
    { u.attempt() } -> std::convertible_to<double>;
    u.accept();
};

/**
 * @ingroup simplemc-mc-kernels
 * @brief Contract a type must satisfy to drive a Monte Carlo simulation as a kernel.
 *
 * @details A kernel is the algorithm that advances the simulation by one MC step. The free
 * simplemc::run loop accepts any type that exposes a callable `step(rng)` member, where `rng` is some
 * random number generator.
 *
 * The default simplemc::metropolis_kernel implements a standard Metropolis-Hastings step over a
 * simplemc::update_set.
 *
 * Users can implement their own custom kernels (parallel tempering, heat-bath, Wolff cluster, ...) by
 * satisfying this concept.
 *
 * Kernels may optionally expose a `prepare()` member. If present, simplemc::run calls it once at the
 * start of a simulation.
 *
 * @tparam K Type to check.
 * @tparam RNG RNG type the kernel will receive.
 */
template <typename K, typename RNG>
concept mc_kernel = requires(K& k, RNG& rng) { k.step(rng); };

/**
 * @ingroup simplemc-mc-callbacks
 * @brief Contract a type must satisfy to be passed as the callback bundle to simplemc::run.
 *
 * @details The run loop supports four callback hooks, each taking the current simulation context
 * `ctx` as an argument (see also simplemc::simulation_ctx). 
 * 
 * Let `c` be an instance of type `C`. The requirements for a type `C` to be a callback bundle are
 *
 * - `c.on_step(ctx)` is a valid expression (called after every kernel step),
 * - `c.on_cycle(ctx)` is a valid expression (called after each cycle's measurement sweep),
 * - `c.on_checkpoint(ctx)` is a valid expression (called when a checkpoint threshold is crossed (see
 * simplemc::simulation_params)), and
 * - `c.stop_when(ctx) -> bool` returns a boolean flag indicating whether to stop the simulation
 * (`false`) or to keep going (`true`). It is polled in the outer loop condition.
 *
 * All four callbacks must be invocable from a non-const bundle (so callbacks may mutate their own
 * state).
 *
 * See simplemc::run_callbacks for more information and defaults.
 *
 * @tparam C Type to check.
 */
template <typename C>
concept mc_run_callbacks = requires(C& c, const simulation_ctx& x) {
    c.on_step(x);
    c.on_cycle(x);
    c.on_checkpoint(x);
    { c.stop_when(x) } -> std::convertible_to<bool>;
};

/**
 * @ingroup simplemc-mc-utils
 * @brief Check if type `T` is serializable by a serializer of type `S` via a call to
 * `simplemc_save_input_config`.
 *
 * @details Parallels simplemc::has_simplemc_save but its intended use is for user-input.
 *
 * @tparam T Type being serialized.
 * @tparam S Serializer type.
 */
template <typename T, typename S>
concept has_simplemc_save_input_config = requires(const T& t, S& s) { simplemc_save_input_config(s, t); };

/**
 * @ingroup simplemc-mc-utils
 * @brief Check if type `T` is deserializable by a serializer of type `S` via a call to
 * `simplemc_load_input_config`.
 *
 * @details Parallels simplemc::has_simplemc_load but its intended use is for user-input.
 *
 * @tparam T Type being deserialized into.
 * @tparam S Serializer type.
 */
template <typename T, typename S>
concept has_simplemc_load_input_config = requires(T& t, const S& s) { simplemc_load_input_config(s, t); };

/**
 * @ingroup simplemc-mc-utils
 * @brief Check if type `T` can be collected across MPI ranks via a call to `simplemc_mpi_collect`.
 *
 * @details The overload must take `(const mpi::communicator&, const T&)` and return a `T`.
 *
 * This is the shape required by the simplemc::basic_measurement / simplemc::basic_update wrapper hook
 * to forward MPI reduction through to the wrapped user value.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept has_simplemc_mpi_collect = requires(const mpi::communicator& c, const T& v) {
    { simplemc_mpi_collect(c, v) } -> std::same_as<T>;
};

} // namespace simplemc

#endif // SIMPLEMC_MC_CONCEPTS_HPP
