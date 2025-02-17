/**
 * @file
 * @brief Random number generator xoshiro256.
 */

#ifndef SIMPLEMC_RANDOM_XOSHIRO256_HPP
#define SIMPLEMC_RANDOM_XOSHIRO256_HPP

#include <simplemc/random/splitmix64.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-random-rngs
 * @{
 */

/**
 * @brief Enumerate different types of the simplemc::xoshiro256 RNG.
 *
 * @details The following types are available:
 * - `plus` (simplemc::xoshiro256p): The fastest %xoshiro256 RNG. Suitable for floating-point number
 * generation.
 * - `plusplus` (simplemc::xoshiro256pp): General-purpose RNG.
 * - `starstar` (simplemc::xoshiro256ss): General-purpose RNG.
 */
enum class xoshiro256_type { plus, plusplus, starstar };

/**
 * @brief %xoshiro256 random number generators for 64-bit unsigned integer values.
 *
 * @details Based on the <a href="http://prng.di.unimi.it/xoshiro256plus.c">C implementation</a> by
 * Sebastiano Vigna.
 *
 * It satisfies the C++ named requirements: <a href="https://en.cppreference.com/w/cpp/named_req/
 * RandomNumberEngine">RandomNumberEngine</a> and can be used for floating point number generation
 * (simplemc::xoshiro256p) as well as general purposes like Monte Carlo simulations
 * (simplemc::xoshiro256pp and simplemc::xoshiro256ss).
 *
 * The internal state is stored in 256 bits, a `std::array<std::uint64_t, 4>`.
 *
 * @code{.cpp}
 * #include <fmt/base.h>
 * #include <simplemc/random.hpp>
 *
 * int main() {
 *     // initialize xoshiro256ss RNG
 *     simplemc::xoshiro256ss rng{};
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
 * 11488543998296168227
 * 8622350652098268912
 * 9422899847795208629
 * 13062981702128504018
 * 6540931487853183745
 * ```
 *
 * @tparam X The simplemc::xoshiro256_type.
 */
template <xoshiro256_type X>
class xoshiro256 {
public:
    /**
     * @brief Type of the internal state.
     */
    using state_type = std::array<std::uint64_t, 4>;

    /**
     * @brief Type of the generated random number.
     */
    using result_type = std::uint64_t;

    /**
     * @brief Default seed for a simplemc::splitmix64 RNG which is used to set the internal state.
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
     * @details The value is used to seed a simplemc::splitmix64 RNG, which in turn is used to set
     * the internal state.
     *
     * @param s 64-bit unsigned integer seed.
     */
    explicit xoshiro256(std::uint64_t s = default_seed) { seed(s); }

    /**
     * @brief Construct an RNG from four `std::uint64_t` values.
     *
     * @details It simply sets the internal state to the given values.
     *
     * @param s0 64-bit unsigned integer.
     * @param s1 64-bit unsigned integer.
     * @param s2 64-bit unsigned integer.
     * @param s3 64-bit unsigned integer.
     */
    xoshiro256(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3) :
        state_(state_type { s0, s1, s2, s3 }) {}

    /**
     * @brief Construct an RNG from a seed sequence.
     *
     * @details It forwards the given seed sequence to seed(SeedSeq&) to set the internal state.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence object.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, xoshiro256<X>> && !std::is_arithmetic_v<SeedSeq>)
    explicit xoshiro256(SeedSeq& seq) {
        seed(seq);
    }

    /**
     * @brief Set the internal state using a single `std::uint64_t` value.
     *
     * @details The value is used to seed a simplemc::splitmix64 RNG, which in turn is used to set
     * the internal state.
     *
     * @param s 64-bit unsigned integer.
     */
    void seed(std::uint64_t s = default_seed) {
        splitmix64 sm(s);
        state_[0] = sm();
        state_[1] = sm();
        state_[2] = sm();
        state_[3] = sm();
    }

    /**
     * @brief Set the internal state to the given four `std::uint64_t` values.
     *
     * @param s0 64-bit unsigned integer.
     * @param s1 64-bit unsigned integer.
     * @param s2 64-bit unsigned integer.
     * @param s3 64-bit unsigned integer.
     */
    void seed(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3) {
        state_ = state_type { s0, s1, s2, s3 };
    }

    /**
     * @brief Set the internal state using a seed sequence.
     *
     * @details It uses the seed sequence to generate eight `std::uint32_t` numbers and writes them
     * into the 256 bits of the internal state.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence object.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, xoshiro256<X>> && !std::is_arithmetic_v<SeedSeq>)
    void seed(SeedSeq& seq) {
        // low level hack to seed the state array
        seq.generate(reinterpret_cast<std::uint32_t*>(state_.data()), // NOLINT (reinterpret_cast is needed here)
            reinterpret_cast<std::uint32_t*>(state_.data() + 4)); // NOLINT (reinterpret_cast is needed here)
    }

    /**
     * @brief Get the current internal state.
     *
     * @return `std::array<std::uint64_t, 4>` representing the internal state.
     */
    [[nodiscard]] const state_type& internal_state() const { return state_; }

    /**
     * @brief Generate a random number and advance the internal state.
     *
     * @return A random 64-bit unsigned number.
     */
    result_type operator()() {
        std::uint64_t result = type_specific();
        std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] ^= rotl(state_[3], 45);
        return result;
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
     * @brief Advance the internal state \f$ 2^{128} \f$ steps.
     */
    void jump() {
        static const state_type arr =
            state_type { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
        base_jump(arr);
    }

    /**
     * @brief Advance the internal state \f$ 2^{192} \f$ steps.
     */
    void long_jump() {
        static const state_type arr =
            state_type { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
        base_jump(arr);
    }

    /**
     * @brief Compare two simplemc::xoshiro256 objects for equality.
     *
     * @param lhs Left hand side RNG.
     * @param rhs Right hand side RNG.
     * @return True if their internal states are equal.
     */
    [[nodiscard]] friend bool operator==(const xoshiro256& lhs, const xoshiro256& rhs) {
        return lhs.state_ == rhs.state_;
    }

    /**
     * @brief Compare two simplemc::xoshiro256 objects for inequality.
     *
     * @param lhs Left hand side RNG.
     * @param rhs Right hand side RNG.
     * @return True if their internal states are distinct.
     */
    [[nodiscard]] friend bool operator!=(const xoshiro256& lhs, const xoshiro256& rhs) { return !(lhs == rhs); }

    /**
     * @brief Write a textual representation of a simplemc::xoshiro256 object to `std::ostream`.
     *
     * @details Throws an exception, if writing to `std::ostream` fails.
     *
     * @param os `std::ostream` to write to.
     * @param eng RNG to be written.
     * @return Reference to the `std::ostream`.
     */
    friend std::ostream& operator<<(std::ostream& os, const xoshiro256& eng) {
        const auto& state = eng.internal_state();
        if (!(os << state[0] << ' ' << state[1] << ' ' << state[2] << ' ' << state[3])) {
            throw simplemc_exception("Error writing a simplemc::xoshiro256 to ostream.");
        }
        return os;
    }

    /**
     * @brief Read a textual representation of a simplemc::xoshiro256 object from `std::istream`.
     *
     * @details Throws an exception, if reading from `std::istream` fails.
     *
     * @param is `std::istream` to read from.
     * @param eng RNG to read into.
     * @return Reference to the `std::istream`.
     */
    friend std::istream& operator>>(std::istream& is, xoshiro256& eng) {
        typename xoshiro256<X>::state_type arr;
        if (is >> arr[0] >> std::ws >> arr[1] >> std::ws >> arr[2] >> std::ws >> arr[3]) {
            eng.seed(arr[0], arr[1], arr[2], arr[3]);
        } else {
            throw simplemc_exception("Error reading a simplemc::xoshiro256 from istream.");
        }
        return is;
    }

private:
    // Perform a left rotation.
    [[nodiscard]] std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    // Operation dependent on the xoshiro256_type.
    [[nodiscard]] std::uint64_t type_specific();

    // Perform the actual jump.
    void base_jump(const state_type& arr) {
        std::uint64_t s0 = 0;
        std::uint64_t s1 = 0;
        std::uint64_t s2 = 0;
        std::uint64_t s3 = 0;
        for (unsigned long long a : arr) {
            for (int b = 0; b < 64; b++) {
                if (a & static_cast<std::uint64_t>(1) << b) {
                    s0 ^= state_[0];
                    s1 ^= state_[1];
                    s2 ^= state_[2];
                    s3 ^= state_[3];
                }
                this->operator()();
            }
        }
        state_[0] = s0;
        state_[1] = s1;
        state_[2] = s2;
        state_[3] = s3;
    }

    state_type state_ {};
};

template <>
inline std::uint64_t xoshiro256<xoshiro256_type::plus>::type_specific() {
    return state_[0] + state_[3];
}

template <>
inline std::uint64_t xoshiro256<xoshiro256_type::plusplus>::type_specific() {
    return rotl(state_[0] + state_[3], 23) + state_[0];
}

template <>
inline std::uint64_t xoshiro256<xoshiro256_type::starstar>::type_specific() {
    return rotl(state_[1] * 5, 7) * 9;
}

/**
 * @brief xoshiro256+ RNG suitable for floating-point number generation.
 */
using xoshiro256p = xoshiro256<xoshiro256_type::plus>;

/**
 * @brief xoshiro256++ RNG for general purposes.
 */
using xoshiro256pp = xoshiro256<xoshiro256_type::plusplus>;

/**
 * @brief xoshiro256** RNG for general purposes.
 */
using xoshiro256ss = xoshiro256<xoshiro256_type::starstar>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_XOSHIRO256_HPP
