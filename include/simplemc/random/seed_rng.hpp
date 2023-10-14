/**
 * @file samples.hpp
 * @brief Utility function to seed an RNG with a seed sequence.
 */

#ifndef SIMPLEMC_RANDOM_SEED_RNG_HPP
#define SIMPLEMC_RANDOM_SEED_RNG_HPP

#include <simplemc/random/splitmix64.hpp>

#include <random>

namespace simplemc {

/**
 * @brief Seed RNG with a seed sequence depending on an input parameter (useful for parallelization).
 *
 * @tparam RNG Random number generator type.
 * @param rng Random number generator.
 * @param rank Rank of the process.
 */
template <typename RNG> 
void seed_rng(RNG& rng, int rank = 0) {
    splitmix64 sm64 { splitmix64::default_seed + 0x2544382c71ac491b * rank };
    std::seed_seq seq { sm64(), sm64(), sm64(), sm64() };
    rng.seed(seq);
}

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SEED_RNG_HPP
