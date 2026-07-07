/**
 * @file
 * @brief Random number generator xoshiro256.
 */

#ifndef SIMPLEMC_RANDOM_XOSHIRO256_HPP
#define SIMPLEMC_RANDOM_XOSHIRO256_HPP

#include <simplemc/random/splitmix64.hpp>
#include <simplemc/serialize/concepts.hpp>
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
 * The internal state is stored in 256 bits, a `std::array<std::uint64_t, 4>`. The period of the
 * generator is \f$ 2^{256} - 1 \f$.
 *
 * @include random/doc_xoshiro256.cpp
 *
 * Output:
 *
 * ```
 * Random numbers in the interval [0, 18446744073709551615]:
 * 11488543998296168227
 * 8622350652098268912
 * 9422899847795208629
 * 14517156874756448551
 * 2588083500738939770
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
     * @return Zero.
     */
    [[nodiscard]] static constexpr result_type min() noexcept { return 0; }

    /**
     * @brief Upper bound of generated random numbers.
     *
     * @return `std::numeric_limits<std::uint64_t>::%max()`.
     */
    [[nodiscard]] static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }

    /**
     * @brief Construct a %xoshiro256 RNG from a single `std::uint64_t` seed.
     *
     * @details The value is used to seed a simplemc::splitmix64 RNG, which in turn is used to set
     * the internal state.
     *
     * @param s 64-bit unsigned integer seed.
     */
    explicit constexpr xoshiro256(std::uint64_t s = default_seed) noexcept { seed(s); }

    /**
     * @brief Construct a %xoshiro256 RNG from four `std::uint64_t` values.
     *
     * @details It simply sets the internal state to the given values.
     *
     * @param s0 64-bit unsigned integer.
     * @param s1 64-bit unsigned integer.
     * @param s2 64-bit unsigned integer.
     * @param s3 64-bit unsigned integer.
     */
    constexpr xoshiro256(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3) noexcept :
        state_ { s0, s1, s2, s3 } {}

    /**
     * @brief Construct a %xoshiro256 RNG from a seed sequence.
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
    constexpr void seed(std::uint64_t s = default_seed) noexcept {
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
    constexpr void seed(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3) noexcept {
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
        seq.generate(reinterpret_cast<std::uint32_t*>(state_.data()), // NOLINT (reinterpret_cast needed)
            reinterpret_cast<std::uint32_t*>(state_.data() + 4) // NOLINT (reinterpret_cast needed)
        );
    }

    /**
     * @brief Get the current internal state.
     *
     * @return Const reference to the internal state array.
     */
    [[nodiscard]] constexpr const state_type& internal_state() const noexcept { return state_; }

    /**
     * @brief Generate a random number and advance the internal state.
     *
     * @return Random 64-bit unsigned integer.
     */
    constexpr result_type operator()() noexcept {
        const std::uint64_t result = type_specific();
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    /**
     * @brief Advance the internal state as if operator()() has been called `z` times.
     *
     * @details This operation has \f$ \mathcal{O}(z) \f$ complexity.
     *
     * For large jumps, consider using jump() or long_jump() instead.
     *
     * @param z Number of steps to jump ahead.
     */
    constexpr void discard(std::uintmax_t z) noexcept {
        for (std::uintmax_t i = 0; i < z; ++i) {
            this->operator()();
        }
    }

    /**
     * @brief Advance the internal state \f$ 2^{128} \f$ steps in \f$ \mathcal{O}(1) \f$ time.
     *
     * @details This is useful for generating non-overlapping subsequences for parallel computation.
     * Each call to jump() is equivalent to \f$ 2^{128} \f$ calls to operator()().
     */
    constexpr void jump() noexcept {
        constexpr state_type arr { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
        base_jump(arr);
    }

    /**
     * @brief Advance the internal state \f$ 2^{192} \f$ steps in \f$ \mathcal{O}(1) \f$ time.
     *
     * @details This is useful for generating non-overlapping subsequences for parallel computation.
     * Each call to long_jump() is equivalent to \f$ 2^{192} \f$ calls to operator()().
     */
    constexpr void long_jump() noexcept {
        constexpr state_type arr { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
        base_jump(arr);
    }

    /**
     * @brief Equality comparison operator compares the internal states of two simplemc::xoshiro256
     * objects.
     *
     * @param lhs Left hand side RNG.
     * @param rhs Right hand side RNG.
     * @return True if their internal states are equal.
     */
    [[nodiscard]] friend constexpr bool operator==(const xoshiro256& lhs, const xoshiro256& rhs) noexcept {
        return lhs.state_ == rhs.state_;
    }

    /**
     * @brief Write a textual representation of a simplemc::xoshiro256 object to `std::ostream`.
     *
     * @details It throws a simplemc::simplemc_exception if writing to `std::ostream` fails.
     *
     * @param os `std::ostream` to write to.
     * @param eng RNG to be written.
     * @return Reference to the `std::ostream`.
     */
    friend std::ostream& operator<<(std::ostream& os, const xoshiro256& eng) {
        const auto& state = eng.internal_state();
        if (!(os << state[0] << ' ' << state[1] << ' ' << state[2] << ' ' << state[3])) {
            throw simplemc_exception("Error writing a simplemc::xoshiro256 to ostream");
        }
        return os;
    }

    /**
     * @brief Read a textual representation of a simplemc::xoshiro256 object from `std::istream`.
     *
     * @details It throws a simplemc::simplemc_exception if reading from `std::istream` fails.
     *
     * @param is `std::istream` to read from.
     * @param eng RNG to read into.
     * @return Reference to the `std::istream`.
     */
    friend std::istream& operator>>(std::istream& is, xoshiro256& eng) {
        state_type arr {};
        if (is >> arr[0] >> std::ws >> arr[1] >> std::ws >> arr[2] >> std::ws >> arr[3]) {
            eng.seed(arr[0], arr[1], arr[2], arr[3]);
        } else {
            throw simplemc_exception("Error reading a simplemc::xoshiro256 from istream");
        }
        return is;
    }

private:
    /// Perform a left rotation.
    [[nodiscard]] static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    /// Operation dependent on the xoshiro256_type.
    [[nodiscard]] constexpr std::uint64_t type_specific() const noexcept;

    /// Perform the actual jump.
    constexpr void base_jump(const state_type& arr) noexcept {
        std::uint64_t s0 = 0;
        std::uint64_t s1 = 0;
        std::uint64_t s2 = 0;
        std::uint64_t s3 = 0;
        for (const auto a : arr) {
            for (int b = 0; b < 64; ++b) {
                if (a & (static_cast<std::uint64_t>(1) << b)) {
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

private:
    state_type state_ {};
};

template <>
[[nodiscard]] inline constexpr std::uint64_t xoshiro256<xoshiro256_type::plus>::type_specific() const noexcept {
    return state_[0] + state_[3];
}

template <>
[[nodiscard]] inline constexpr std::uint64_t xoshiro256<xoshiro256_type::plusplus>::type_specific() const noexcept {
    return rotl(state_[0] + state_[3], 23) + state_[0];
}

template <>
[[nodiscard]] inline constexpr std::uint64_t xoshiro256<xoshiro256_type::starstar>::type_specific() const noexcept {
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

/**
 * @brief Serialize a simplemc::xoshiro256.
 *
 * @details It serializes the four 64-bit state words of the RNG.
 *
 * @tparam S simplemc::serializer type.
 * @tparam X simplemc::xoshiro256_type variant of the RNG.
 * @param s Serializer object.
 * @param r RNG to serialize.
 */
template <serializer S, xoshiro256_type X>
void simplemc_save(S s, const xoshiro256<X>& r) {
    const auto& st = r.internal_state();
    s.save_at("s0", st[0]);
    s.save_at("s1", st[1]);
    s.save_at("s2", st[2]);
    s.save_at("s3", st[3]);
}

/**
 * @brief Deserialize a simplemc::xoshiro256.
 *
 * @details It deserializes the four 64-bit state words and uses them to seed the RNG (see
 * simplemc::xoshiro256::seed).
 *
 * @tparam S simplemc::serializer type.
 * @tparam X simplemc::xoshiro256_type variant of the RNG.
 * @param s Serializer object.
 * @param r RNG to deserialize into.
 */
template <serializer S, xoshiro256_type X>
void simplemc_load(const S& s, xoshiro256<X>& r) {
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    s.load_at("s0", s0);
    s.load_at("s1", s1);
    s.load_at("s2", s2);
    s.load_at("s3", s3);
    r.seed(s0, s1, s2, s3);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_XOSHIRO256_HPP
