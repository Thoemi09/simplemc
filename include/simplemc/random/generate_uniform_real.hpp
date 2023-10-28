/**
 * @file generate_uniform_real.hpp
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
 * @tparam Engine 64-bit RNG.
 * @param eng RNG.
 * @param min Lower bound.
 * @param max Upper bound.
 * @return Double value on the specified interval.
 */
template <typename Engine>
    requires std::is_same_v<typename Engine::result_type, std::uint64_t>
inline double generate_uniform_real(Engine& eng, double min, double max) {
    return min + (max - min) * ((eng() >> 11) * 0x1.0p-53);
}

} // namespace simplemc::detail

#endif // SIMPLEMC_RANDOM_GENERATE_UNIFORM_REAL_HPP
