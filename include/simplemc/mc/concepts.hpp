/**
 * @file
 * @brief Concepts that describe the user-facing contracts of simplemc-mc.
 */

#ifndef SIMPLEMC_MC_CONCEPTS_HPP
#define SIMPLEMC_MC_CONCEPTS_HPP

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

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_CONCEPTS_HPP
