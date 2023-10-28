/**
 * @file concepts.hpp
 * @brief Concepts and type traits for the simplemc library.
 */

#ifndef SIMPLEMC_UTILS_CONCEPTS_HPP
#define SIMPLEMC_UTILS_CONCEPTS_HPP

#include <concepts>

namespace simplemc {

/**
 * @brief A concept that checks if a type is any of the given types. Taken from cppreference.com.
 * 
 * @tparam T The type to check.
 * @tparam U The types to compare against.
 * @return True if T is any of the types in U, false otherwise.
 */
template <typename T, typename... U>
concept is_any_of = (std::same_as<T, U> || ...);


/**
 * @brief A concept that checks if a type is an integer type except for bool, char, and wchar_t.
 * 
 * @tparam T The type to check.
 * @return True if T is an accepted integer type, false otherwise.
 */
template <typename T>
concept integer_only = is_any_of<T, short, int, long, long long, unsigned short, unsigned int, unsigned long, unsigned long long>;

/**
 * @brief Indicating column-major order.
 */
struct column_major {};

/**
 * @brief Indicating row-major order.
 */
struct row_major {};

/**
 * @brief Concept describing the order of a multi-dimensional array.
 */
template <typename T>
concept nd_order = is_any_of<T, column_major, row_major>;

} // namespace simplemc

#endif // SIMPLEMC_UTILS_CONCEPTS_HPP
