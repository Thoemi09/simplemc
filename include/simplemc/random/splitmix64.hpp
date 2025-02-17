/**
 * @file
 * @brief Random number generator splitmix64.
 */

#ifndef SIMPLEMC_RANDOM_SPLITMIX64_HPP
#define SIMPLEMC_RANDOM_SPLITMIX64_HPP

#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace simplemc {

/**
 * @ingroup simplemc-random-rngs
 * @brief Fast random number generator for 64-bit unsigned integer values.
 *
 * @details Based on the <a href="http://prng.di.unimi.it/splitmix64.c">C implementation</a> by
 * Sebastiano Vigna.
 *
 * It satisfies the C++ named requirements: <a href="https://en.cppreference.com/w/cpp/named_req/
 * RandomNumberEngine">RandomNumberEngine</a> and can be useful for seeding more sophisticated
 * engines.
 *
 * The internal state is a 64 bits, a single `std::uint64_t`.
 *
 * @code{.cpp}
 * #include <fmt/base.h>
 * #include <simplemc/random.hpp>
 *
 * int main() {
 *     // initialize splitmix64 RNG
 *     simplemc::splitmix64 rng{};
 *
 *     // print 5 random numbers
 *     fmt::println("Random numbers in the interval [{}, {}]:", rng.min(), rng.max());
 *     for (int i = 0; i < 5; ++i) {
 *         fmt::println("{}", rng());
 *     }
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Random numbers in the interval [0, 18446744073709551615]:
 * 1719758671208677865
 * 13238173923722200173
 * 13844957877086618154
 * 16939971842163881215
 * 15293509828263232410
 * ```
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
     * @brief Default seed for the internal state.
     */
    static constexpr std::uint64_t default_seed = 0x8a34e2345234fdb1;

    /**
     * @brief Lower bound of generated random numbers.
     *
     * @returns Zero.
     */
    [[nodiscard]] static constexpr result_type min() { return 0; }

    /**
     * @brief Upper bound of generated random numbers.
     *
     * @returns `std::numeric_limits<std::uint64_t>::%max()`.
     */
    [[nodiscard]] static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    /**
     * @brief Construct an RNG from a single `std::uint64_t` seed.
     *
     * @details It simply sets the internal state to the given seed.
     *
     * @param s 64-bit unsigned integer seed.
     */
    explicit splitmix64(std::uint64_t s = default_seed) : state_(s) {}

    /**
     * @brief Construct an RNG from a seed sequence.
     *
     * @details It forwards the given seed sequence to seed(SeedSeq&) to set the internal state.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence object.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, splitmix64> && !std::is_arithmetic_v<SeedSeq>)
    explicit splitmix64(SeedSeq& seq) {
        seed(seq);
    }

    /**
     * @brief Set the internal state to the given `std::uint64_t`.
     *
     * @param s 64-bit unsigned integer.
     */
    void seed(std::uint64_t s = default_seed) { state_ = s; }

    /**
     * @brief Set the internal state using the given seed sequence.
     *
     * @details It uses the seed sequence to generate two `std::uint32_t` numbers and combines them to
     * form a new 64-bit internal state by simply joining their bits.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence object.
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
     * @return `std::uint64_t` representing the internal state.
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
     * @brief Advance the internal state as if the operator()() has been called `z` times.
     *
     * @param z Number of steps to jump ahead.
     */
    void discard(std::uintmax_t z) {
        for (std::uintmax_t i = 0; i < z; i++) {
            this->operator()();
        }
    }

    /**
     * @brief Compare two simplemc::splitmix64 objects for equality.
     *
     * @param lhs Left hand side RNG.
     * @param rhs Right hand side RNG.
     * @return True if their internal states are equal.
     */
    [[nodiscard]] friend bool operator==(const splitmix64& lhs, const splitmix64& rhs) {
        return lhs.state_ == rhs.state_;
    }

    /**
     * @brief Compare two simplemc::splitmix64 objects for inequality.
     *
     * @param lhs Left hand side RNG.
     * @param rhs Right hand side RNG.
     * @return True if their internal states are distinct.
     */
    [[nodiscard]] friend bool operator!=(const splitmix64& lhs, const splitmix64& rhs) { return !(lhs == rhs); }

    /**
     * @brief Write a textual representation of a simplemc::splitmix64 object to `std::ostream`.
     *
     * @details Throws an exception, if writing to `std::ostream` fails.
     *
     * @param os `std::ostream` to write to.
     * @param eng RNG to be written.
     * @return Reference to the `std::ostream`.
     */
    friend std::ostream& operator<<(std::ostream& os, const splitmix64& eng) {
        if (!(os << eng.internal_state())) {
            throw simplemc_exception("Error writing a simplemc::splitmix64 to ostream");
        }
        return os;
    }

    /**
     * @brief Read a textual representation of a simplemc::splitmix64 object from `std::istream`.
     *
     * @details Throws an exception, if reading from `std::istream` fails.
     *
     * @param is `std::istream` to read from.
     * @param eng RNG to read into.
     * @return Reference to the `std::istream`.
     */
    friend std::istream& operator>>(std::istream& is, splitmix64& eng) {
        std::uint64_t s {};
        if (is >> s) {
            eng.seed(s);
        } else {
            throw simplemc_exception("Error reading a simplemc::splitmix64 from istream");
        }
        return is;
    }

private:
    std::uint64_t state_ { default_seed };
};

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SPLITMIX64_HPP
