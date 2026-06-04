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
 * once per measurement cycle. The contract is intentionally minimal: a single `measure()` member
 * function callable on a mutable instance.
 *
 * @tparam T Type to check.
 */
template <class T>
concept mc_measurement = requires(T& m) { m.measure(); };

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_CONCEPTS_HPP
