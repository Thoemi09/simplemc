/**
 * @file
 * @brief General concepts for the simplemc library.
 */

#ifndef SIMPLEMC_UTILS_CONCEPTS_HPP
#define SIMPLEMC_UTILS_CONCEPTS_HPP

#include <complex>
#include <concepts>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-concepts
 * @{
 */

/**
 * @brief A concept that checks if a type is any of the given types.
 *
 * @details Taken from <a href="https://en.cppreference.com/w/cpp/concepts/same_as">cppreference</a>.
 *
 * @tparam T Type to check.
 * @tparam Us Types to compare against.
 */
template <typename T, typename... Us>
concept is_any_of = (std::same_as<T, Us> || ...);

/**
 * @brief A concept that checks if a type is an integer type except for `bool`, `char`, and `wchar_t`.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept integer_only =
    is_any_of<T, short, int, long, long long, unsigned short, unsigned int, unsigned long, unsigned long long>;

/**
 * @brief A concept that checks if a type is either a `double` or a `std::complex<double>`.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept double_or_complex = is_any_of<T, double, std::complex<double>>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_CONCEPTS_HPP
