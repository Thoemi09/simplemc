// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Utility function to seed an RNG with a seed sequence.
 */

#ifndef SIMPLEMC_RANDOM_SEED_RNG_HPP
#define SIMPLEMC_RANDOM_SEED_RNG_HPP

#include <simplemc/random/splitmix64.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-random-rngs
 * @{
 */

/**
 * @brief Seed a RNG with a `std::seed_seq` depending on a base seed and an MPI rank.
 *
 * @details The base seed and the rank of the calling process are mixed to construct a
 * simplemc::splitmix64 RNG, which in turn is used to initialize a `std::seed_seq` object. The seed
 * sequence is then passed to the RNG object to seed it.
 *
 * It is intended to be used in MPI applications where each process is supposed to have an independent
 * RNG. Passing a user-provided base seed (e.g. read from an input-config file) yields deterministic
 * but independent per-rank streams:
 *
 * @code{.cpp}
 * #include <simplemc/mpi.hpp>
 * #include <simplemc/random.hpp>
 *
 * int main(int argc, char** argv) {
 *     // initialize MPI environment
 *     simplemc::mpi::environment env(argc, argv);
 *     simplemc::mpi::communicator comm;
 *
 *     // construct and seed random number generator (seed typically read from an input-config file)
 *     simplemc::xoshiro256pp rng;
 *     simplemc::seed_rng(rng, comm.rank(), 0xc0ffee);
 *
 *     // use the RNG ...
 * }
 * @endcode
 *
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param rank Rank of the process.
 * @param seed Base seed mixed with the rank.
 * @param num Number of integers consumed by the `std::seed_seq`.
 */
template <typename RNG>
void seed_rng(RNG& rng, int rank = 0, std::uint64_t seed = splitmix64::default_seed, std::size_t num = 4) {
    splitmix64 sm64 { seed + 0x2544382c71ac491b * static_cast<std::uint64_t>(rank) };
    std::vector<splitmix64::result_type> ints(num);
    for (auto& i : ints) {
        i = sm64();
    }
    std::seed_seq seq(ints.begin(), ints.end());
    rng.seed(seq);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SEED_RNG_HPP
