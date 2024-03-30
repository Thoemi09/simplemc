/**
 * @file
 * @brief Concepts and type traits for the simplemc library.
 */

#ifndef SIMPLEMC_UTILS_CONCEPTS_HPP
#define SIMPLEMC_UTILS_CONCEPTS_HPP

#include <complex>
#include <concepts>

namespace simplemc {

/**
 * @brief A concept that checks if a type is any of the given types.
 *
 * @details Taken from cppreference.com.
 *
 * @tparam T Type to check.
 * @tparam Us Types to compare against.
 */
template <typename T, typename... Us>
concept is_any_of = (std::same_as<T, Us> || ...);

/**
 * @brief A concept that checks if a type is an integer type except for bool, char, and wchar_t.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept integer_only =
    is_any_of<T, short, int, long, long long, unsigned short, unsigned int, unsigned long, unsigned long long>;

/**
 * @brief A concept that checks if a type is either a double or a std::complex<double>.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept double_or_complex = is_any_of<T, double, std::complex<double>>;

/**
 * @brief Tag indicating column-major order.
 */
struct column_major {};

/**
 * @brief Tag indicating row-major order.
 */
struct row_major {};

/**
 * @brief A concept that checks if a given type is either a column_major or a row_major tag.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept nd_order = is_any_of<T, column_major, row_major>;

} // namespace simplemc

#endif // SIMPLEMC_UTILS_CONCEPTS_HPP
