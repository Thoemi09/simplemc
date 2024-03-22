/**
 * @file
 * @brief Random number generator splitmix64.
 */

#ifndef SIMPLEMC_RANDOM_SPLITMIX64_HPP
#define SIMPLEMC_RANDOM_SPLITMIX64_HPP

#include <array>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <type_traits>

namespace simplemc {

/**
 * @brief Fast random number generator for 64 bit unsigned integer values.
 *
 * @details Based on the C implementation by Sebastiano Vigna (http://prng.di.unimi.it/splitmix64.c).
 * It satisfies the C++ requirements for a RandomNumberEngine and can be useful for seeding more
 * sophisticated engines. The internal state is a single std::uint64_t.
 */
class splitmix64 {
public:
    /**
     * @brief Type of the internal state.
     */
    using state_type = std::uint64_t;

    /**
     * @brief Type of the generated random number.
     */
    using result_type = std::uint64_t;

    /**
     * @brief Default seed.
     */
    static constexpr std::uint64_t default_seed = 0x8a34e2345234fdb1;

    /**
     * @brief Lower bound of generated random numbers.
     */
    [[nodiscard]] static constexpr result_type min() { return 0; }

    /**
     * @brief Upper bound of generated random numbers.
     */
    [[nodiscard]] static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    /**
     * @brief Construct a RNG from a single std::uint64_t seed.
     *
     * @param s 64-bit unsigned integer seed.
     */
    explicit splitmix64(std::uint64_t s = default_seed);

    /**
     * @brief Construct a RNG from a SeedSequence.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, splitmix64> && !std::is_arithmetic_v<SeedSeq>)
    explicit splitmix64(SeedSeq& seq) {
        seed(seq);
    }

    /**
     * @brief Set the internal state to the given std::uint64_t.
     *
     * @param s 64-bit unsigned integer.
     */
    void seed(std::uint64_t s = default_seed) { state_ = s; }

    /**
     * @brief Set the internal state using the given seed sequence.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, splitmix64> && !std::is_arithmetic_v<SeedSeq>)
    void seed(SeedSeq& seq) {
        // hack to fill 64-bits with two 32-bit numbers
        std::array<std::uint32_t, 2> seeds {};
        seq.generate(seeds.begin(), seeds.end());
        state_ = static_cast<std::uint64_t>(seeds[0]) << 32;
        state_ += seeds[1];
    }

    /**
     * @brief Get the current internal state.
     *
     * @return Internal state.
     */
    [[nodiscard]] std::uint64_t internal_state() const { return state_; }

    /**
     * @brief Generate a random number and advance the internal state.
     *
     * @return Random 64-bit unsigned integer.
     */
    result_type operator()() {
        result_type z = (state_ += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    /**
     * @brief Advance the internal state as if the operator() has been called z times.
     *
     * @param z Number of steps to jump ahead.
     */
    void discard(std::uintmax_t z);

    /* Friend declarations. */
    friend std::ostream& operator<<(std::ostream&, const splitmix64&);
    friend std::istream& operator>>(std::istream&, splitmix64&);
    friend bool operator==(const splitmix64&, const splitmix64&);
    friend bool operator!=(const splitmix64&, const splitmix64&);

private:
    std::uint64_t state_ { default_seed };
};

/**
 * @brief Write a textual representation of a random number generator to std::ostream.
 *
 * @details Throws an exception, if writing to std::ostream fails.
 *
 * @param os std::ostream to write to.
 * @param eng RNG to be written.
 * @return Reference to the std::ostream.
 */
std::ostream& operator<<(std::ostream& os, const splitmix64& eng);

/**
 * @brief Read a textual representation of a random number generator from std::istream.
 *
 * @details Throws an exception, if reading from std::istream fails.
 *
 * @param is std::istream to read from.
 * @param eng RNG to read into.
 * @return Reference to the std::istream.
 */
std::istream& operator>>(std::istream& is, splitmix64& eng);

/**
 * @brief Compare two RNGs for equalitiy.
 *
 * @param lhs Left-hand side RNG.
 * @param rhs Right-hand side RNG.
 * @return True if their internal states are equal.
 */
[[nodiscard]] bool operator==(const splitmix64& lhs, const splitmix64& rhs);

/**
 * @brief Compare two RNGs for inequality.
 *
 * @param lhs Left-hand side RNG.
 * @param rhs Right-hand side RNG.
 * @return True if their internal states are distinct.
 */
[[nodiscard]] bool operator!=(const splitmix64& lhs, const splitmix64& rhs);

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SPLITMIX64_HPP
