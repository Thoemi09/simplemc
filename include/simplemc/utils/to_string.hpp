/**
 * @file
 * @brief Get a string representation for types with an overloaded `operator<<`.
 */

#ifndef SIMPLEMC_UTILS_TO_STRING_HPP
#define SIMPLEMC_UTILS_TO_STRING_HPP

#include <sstream>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-general
 * @{
 */

/**
 * @brief Generic function that uses the stream `operator<<` to convert an object to a string.
 *
 * @details It simply streams into a `std::stringstream` and returns the resulting string.
 *
 * @param t Object to convert to a string.
 * @return String representation of the object.
 */
template <typename T>
[[nodiscard]] auto to_string(const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_TO_STRING_HPP
