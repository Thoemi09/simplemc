/**
 * @file
 * @brief Include the correct ranges header files.
 */

#ifndef SIMPLEMC_UTILS_RANGES_HPP
#define SIMPLEMC_UTILS_RANGES_HPP

#include <simplemc/config.hpp>

#ifdef SIMPLEMC_USE_STD_RANGES
#include <ranges>

namespace simplemc {

/**
 * @brief Alias for `std::ranges` to make it interchangeable with the range-v3 library.
 */
namespace ranges = std::ranges;

} // namespace simplemc

#else

#include <range/v3/all.hpp>

#endif // SIMPLMEC_USE_STD_RANGES

#endif // SIMPLEMC_UTILS_RANGES_HPP
