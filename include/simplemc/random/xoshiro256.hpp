/**
 * @file xoshiro256.hpp
 * @brief Random number generator xoshiro256.
 */

#ifndef SIMPLEMC_RANDOM_XOSHIRO256_HPP
#define SIMPLEMC_RANDOM_XOSHIRO256_HPP

#include <array>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <type_traits>

namespace simplemc {

/**
 * @brief Enumerate different types of the xoshiro256 RNG.
 */
enum class xoshiro256_type { plus, plusplus, starstar };

/* Forward declarations. */
template <xoshiro256_type X>
class xoshiro256;
template <xoshiro256_type X>
std::ostream& operator<<(std::ostream&, const xoshiro256<X>&);
template <xoshiro256_type X>
std::istream& operator>>(std::istream&, xoshiro256<X>&);
template <xoshiro256_type X>
bool operator==(const xoshiro256<X>&, const xoshiro256<X>&);
template <xoshiro256_type X>
bool operator!=(const xoshiro256<X>&, const xoshiro256<X>&);

/**
 * @brief Xoshiro256 random number generators.
 *
 * @details Based on the C implementation by Sebastiano Vigna (http://prng.di.unimi.it/xoshiro256plus.c).
 * It satisfies the C++ requirements for a RandomNumberEngine and can be used for floating point number
 * generation (xoshiro256+) as well as general purposes (xoshiro256++/xoshiro256**). The internal state
 * is stored in 256 bits.
 *
 * @tparam X Type of xoshiro256.
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
     * @brief Default seed.
     */
    static constexpr std::uint64_t default_seed = 0x8a34e2345234fdb1;

    /**
     * @brief Lower bound of generated random numbers.
     */
    static constexpr result_type min() { return 0; }

    /**
     * @brief Upper bound of generated random numbers.
     */
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    /**
     * @brief Construct a xoshiro256 RNG from a single std::uint64_t seed.
     *
     * @param s 64-bit unsigned integer seed.
     */
    explicit xoshiro256(std::uint64_t s = default_seed);

    /**
     * @brief Construct a xoshiro256 RNG from four std::uint64_t.
     *
     * @param s0 64-bit unsigned integer.
     * @param s1 64-bit unsigned integer.
     * @param s2 64-bit unsigned integer.
     * @param s3 64-bit unsigned integer.
     */
    xoshiro256(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3);

    /**
     * @brief Construct a xoshiro256 RNG from a SeedSequence.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, xoshiro256<X>> && !std::is_arithmetic_v<SeedSeq>)
    explicit xoshiro256(SeedSeq& seq) {
        seed(seq);
    }

    /**
     * @brief Set the internal state using a single std::uint64_t. The value is used to seed a
     * splitmix64 RNG, which in turn is used to seed the internal state of the xoshiro256 RNG.
     *
     * @param s 64-bit unsigned integer.
     */
    void seed(std::uint64_t s = default_seed);

    /**
     * @brief Set the internal state with four std::uint64_t directly.
     *
     * @param s0 64-bit unsigned integer.
     * @param s1 64-bit unsigned integer.
     * @param s2 64-bit unsigned integer.
     * @param s3 64-bit unsigned integer.
     */
    void seed(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3);

    /**
     * @brief Set the internal state using a SeedSequence.
     *
     * @tparam SeedSeq Type of the seed sequence.
     * @param seq Seed sequence.
     */
    template <typename SeedSeq>
        requires(!std::is_same_v<SeedSeq, xoshiro256<X>> && !std::is_arithmetic_v<SeedSeq>)
    void seed(SeedSeq& seq) {
        // low level hack to seed the state array
        seq.generate(
            reinterpret_cast<std::uint32_t*>(state_.data()), reinterpret_cast<std::uint32_t*>(state_.data() + 4));
    }

    /**
     * @brief Get the current internal state.
     *
     * @return Internal state.
     */
    [[nodiscard]] const state_type& internal_state() const { return state_; }

    /**
     * @brief Generate a random number and advance internal state.
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
     * @brief Advance the internal state as if the operator() has been called z times.
     *
     * @param z Number of steps to jump ahead.
     */
    void discard(std::uintmax_t z);

    /**
     * @brief Advance the internal state 2^128 steps.
     */
    void jump();

    /**
     * @brief Advance the internal state 2^192 steps.
     */
    void long_jump();

    /* Friend declarations. */
    friend std::ostream& operator<< <X>(std::ostream&, const xoshiro256&);
    friend std::istream& operator>> <X>(std::istream&, xoshiro256&);
    friend bool operator== <X>(const xoshiro256&, const xoshiro256&);
    friend bool operator!= <X>(const xoshiro256&, const xoshiro256&);

private:
    /**
     * @brief Perform a left rotation.
     */
    std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    /**
     * @brief Operation dependent on the xoshiro256_type.
     */
    std::uint64_t type_specific();

    /**
     * @brief Perform the actual jump.
     */
    void base_jump(const state_type& arr);

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
 * @brief Write textual representation of splitmix64 to ostream. Throws an exception, if writing to
 * ostream fails.
 *
 * @tparam X xoshiro256 type.
 * @param os Reference to ostream.
 * @param eng xoshiro256 RNG.
 * @return Reference to ostream.
 */
template <xoshiro256_type X>
std::ostream& operator<<(std::ostream& os, const xoshiro256<X>& eng);

/**
 * @brief Restore splitmix64 from istream. Throws an exception, if reading from istream fails.
 *
 * @tparam X xoshiro256 type.
 * @param is Reference to istream.
 * @param eng xoshiro256 RNG.
 * @return Reference to istream.
 */
template <xoshiro256_type X>
std::istream& operator>>(std::istream& is, xoshiro256<X>& eng);

/**
 * @brief Compare two xoshiro256 RNGs for equality.
 *
 * @tparam X xoshiro256 type.
 * @param lhs xoshiro256 #1
 * @param rhs xoshiro256 #2
 * @return True if the internal states are equal.
 */
template <xoshiro256_type X>
bool operator==(const xoshiro256<X>& lhs, const xoshiro256<X>& rhs);

/**
 * @brief Compare two xoshiro256 RNGs for inequality.
 *
 * @tparam X xoshiro256 type.
 * @param lhs xoshiro256 #1
 * @param rhs xoshiro256 #2
 * @return True if the internal states are unequal.
 */
template <xoshiro256_type X>
bool operator!=(const xoshiro256<X>& lhs, const xoshiro256<X>& rhs);

/**
 * @brief xoshiro256+. Use only for floating-point generation.
 */
using xoshiro256p = xoshiro256<xoshiro256_type::plus>;

/**
 * @brief xoshiro256++. Use for all purposes.
 */
using xoshiro256pp = xoshiro256<xoshiro256_type::plusplus>;

/**
 * @brief xoshiro256**. Use for all purposes.
 */
using xoshiro256ss = xoshiro256<xoshiro256_type::starstar>;

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_XOSHIRO256_HPP
