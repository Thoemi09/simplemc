/**
 * @file
 * @brief Generate a random uniform real number.
 */

#ifndef SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP
#define SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP

#include <cstdint>
#include <type_traits>

namespace simplemc::detail {

// Generate a double value on the interval [min, max) given a random 64-bit unsigned integer RNG
// (see http://prng.di.unimi.it/).
template <typename RNG>
    requires std::is_same_v<typename RNG::result_type, std::uint64_t>
[[nodiscard]] inline double generate_uniform_real(RNG& rng, double min, double max) {
    return min + (max - min) * ((rng() >> 11) * 0x1.0p-53);
}

} // namespace simplemc::detail

#endif // SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP
