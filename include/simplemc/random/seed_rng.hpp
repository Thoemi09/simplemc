/**
 * @file
 * @brief Utility function to seed an RNG with a seed sequence.
 */

#ifndef SIMPLEMC_RANDOM_SEED_RNG_HPP
#define SIMPLEMC_RANDOM_SEED_RNG_HPP

#include <simplemc/random/splitmix64.hpp>

#include <cstddef>
#include <random>
#include <vector>

namespace simplemc {

/**
 * @brief Seed RNG with a seed sequence depending on an input parameter (useful for parallelization).
 *
 * @tparam RNG Random number generator.
 * @param rng RNG object.
 * @param rank Rank of the process.
 * @param num Number of integers consumed by the std::seed_seq.
 */
template <typename RNG>
void seed_rng(RNG& rng, int rank = 0, std::size_t num = 4) {
    splitmix64 sm64 { splitmix64::default_seed + 0x2544382c71ac491b * rank };
    std::vector<splitmix64::result_type> ints(num);
    for (auto& i : ints) {
        i = sm64();
    }
    std::seed_seq seq(ints.begin(), ints.end());
    rng.seed(seq);
}

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SEED_RNG_HPP
