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
 * @ingroup simplemc-random-rngs
 * @brief Seed a RNG with a `std::seeq_seq` depending on its MPI rank.
 *
 * @details The rank of the calling process is used to construct a simplemc::splitmix64 RNG, which in
 * turn is used to initialize a `std::seed_seq` object. The seed sequence is then passed to the RNG
 * object to seed it.
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
 *     // construct and seed random number generator
 *     simplemc::xoshiro256pp rng;
 *     simplemc::seed_rng(rng, comm.rank());
 *
 *     // use the RNG ...
 * }
 * @endcode
 *
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param rank Rank of the process.
 * @param num Number of integers consumed by the `std::seed_seq`.
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
