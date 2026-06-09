/**
 * @file
 * @brief Concepts that describe the user-facing contracts of simplemc-mc.
 */

#ifndef SIMPLEMC_MC_CONCEPTS_HPP
#define SIMPLEMC_MC_CONCEPTS_HPP

#include <simplemc/mpi/communicator.hpp>

#include <concepts>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Contract a type must satisfy to be wrapped in a simplemc::measurement.
 *
 * @details A Monte Carlo measurement is an observer of the simulation state that the driver invokes
 * once per measurement cycle.
 *
 * Let `m` be an instance of type `M`. The requirements for a type `M` to be a measurement are the
 * following:
 *
 * - `m.measure` is a valid expression that performs the measurement when invoked.
 *
 * @tparam M Type to check.
 */
template <class M>
concept mc_measurement = requires(M& m) { m.measure(); };

/**
 * @brief Contract a type must satisfy to be wrapped in a simplemc::update.
 *
 * @details A Monte Carlo update proposes a change to the simulation state via `attempt()`, which
 * returns the acceptance probability of the proposed change. The driver then either commits the
 * change by calling `accept()` or rolls it back by calling `reject()` (the latter is an optional
 * callback — see simplemc::update for details).
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
template <class U>
concept mc_update = requires(U& u) {
    { u.attempt() } -> std::convertible_to<double>;
    u.accept();
};

/**
 * @brief Contract a type must satisfy to drive a Monte Carlo run as a kernel.
 *
 * @details A kernel is the algorithm that advances the simulation by one step. The free
 * simplemc::run loop accepts any type that exposes a callable `step(rng)` member, where the RNG
 * type matches the one threaded through the loop. The default simplemc::metropolis_kernel
 * implements a standard Metropolis step over a simplemc::update_set; users can plug in custom
 * kernels (parallel tempering, heat-bath, Wolff cluster, ...) by satisfying this concept.
 *
 * `step` returns `void`. The kernel is responsible for any per-update counter bookkeeping; the
 * outer loop only tracks the global step count and the wall-clock budget.
 *
 * Kernels MAY optionally expose a `prepare()` member; if present, simplemc::run calls it once at
 * the start of a run (used by the default kernel to rebuild the discrete distribution).
 *
 * @tparam K Type to check.
 * @tparam RNG RNG type the kernel will receive.
 */
template <class K, class RNG>
concept mc_kernel = requires(K& k, RNG& rng) { k.step(rng); };

/**
 * @brief Check if type `T` is serializable by a serializer of type `S` via a call to
 * `simplemc_save_input_config`.
 *
 * @details Parallels simplemc::has_simplemc_save but its intended use is for user-input.
 *
 * @tparam T Type being serialized.
 * @tparam S Serializer type.
 */
template <class T, class S>
concept has_simplemc_save_input_config = requires(const T& t, S& s) { simplemc_save_input_config(s, t); };

/**
 * @brief Check if type `T` is deserializable by a serializer of type `S` via a call to
 * `simplemc_load_input_config`.
 *
 * @details Parallels simplemc::has_simplemc_load but its intended use is for user-input.
 *
 * @tparam T Type being deserialized into.
 * @tparam S Serializer type.
 */
template <class T, class S>
concept has_simplemc_load_input_config = requires(T& t, const S& s) { simplemc_load_input_config(s, t); };

/**
 * @brief Check if type `T` can be collected across MPI ranks via a call to `simplemc_mpi_collect`.
 *
 * @details The overload must take `(const mpi::communicator&, const T&)` and return a `T`. This is
 * the shape required by the simplemc::basic_measurement / simplemc::basic_update wrapper hook to
 * forward MPI reduction through to the wrapped user value.
 *
 * @tparam T Type to check.
 */
template <class T>
concept has_simplemc_mpi_collect = requires(const mpi::communicator& c, const T& v) {
    { simplemc_mpi_collect(c, v) } -> std::same_as<T>;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_CONCEPTS_HPP
