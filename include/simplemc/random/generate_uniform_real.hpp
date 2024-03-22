/**
 * @file
 * @brief Generate a uniform real number.
 */

#ifndef SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP
#define SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP

#include <cstdint>
#include <type_traits>

namespace simplemc::detail {

/**
 * @brief Generate a double value on the interval [min, max) given a random
 * 64-bit unsigned integer RNG (see http://prng.di.unimi.it/).
 *
 * @tparam RNG 64-bit random number generator.
 * @param rng RNG object.
 * @param min Lower bound.
 * @param max Upper bound.
 * @return Double value on the specified interval.
 */
template <typename RNG>
    requires std::is_same_v<typename RNG::result_type, std::uint64_t>
[[nodiscard]] inline double generate_uniform_real(RNG& rng, double min, double max) {
    return min + (max - min) * ((rng() >> 11) * 0x1.0p-53);
}

} // namespace simplemc::detail

#endif // SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP
