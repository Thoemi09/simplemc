// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief General type traits for the simplemc library.
 */

#ifndef SIMPLEMC_UTILS_TRAITS_HPP
#define SIMPLEMC_UTILS_TRAITS_HPP

#include <complex>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-traits
 * @{
 */

/**
 * @brief Trait detecting whether `T` is a `std::complex` of an arithmetic scalar.
 *
 * @tparam T Type to test.
 */
template <typename T>
inline constexpr bool is_std_complex_v = false;

/**
 * @brief Specialization recognizing `std::complex<T>` for arithmetic `T`.
 *
 * @tparam T Complex scalar type.
 */
template <typename T>
inline constexpr bool is_std_complex_v<std::complex<T>> = std::is_arithmetic_v<T>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_TRAITS_HPP
