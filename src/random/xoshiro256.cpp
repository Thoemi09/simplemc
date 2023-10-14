/**
 * @file xoshiro256.cpp
 * @brief Random number generator xoshiro256.
 */

#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <iostream>

namespace simplemc {

template <xoshiro256_type X>
xoshiro256<X>::xoshiro256(std::uint64_t s) {
    seed(s);
}

template <xoshiro256_type X>
xoshiro256<X>::xoshiro256(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3) :
    state_(state_type { s0, s1, s2, s3 }) {}

template <xoshiro256_type X>
void xoshiro256<X>::seed(std::uint64_t s) {
    splitmix64 sm(s);
    state_[0] = sm();
    state_[1] = sm();
    state_[2] = sm();
    state_[3] = sm();
}

template <xoshiro256_type X>
void xoshiro256<X>::seed(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2, std::uint64_t s3) {
    state_ = state_type { s0, s1, s2, s3 };
}

template <xoshiro256_type X>
void xoshiro256<X>::discard(std::uintmax_t z) {
    for (std::uintmax_t i = 0; i < z; i++) {
        this->operator()();
    }
}

template <xoshiro256_type X>
void xoshiro256<X>::jump() {
    static const state_type arr =
        state_type { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
    base_jump(arr);
}

template <xoshiro256_type X>
void xoshiro256<X>::long_jump() {
    static const state_type arr =
        state_type { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
    base_jump(arr);
}

template <xoshiro256_type X>
void xoshiro256<X>::base_jump(const state_type& arr) {
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    for (std::size_t i = 0; i < arr.size(); i++) {
        for (int b = 0; b < 64; b++) {
            if (arr[i] & static_cast<std::uint64_t>(1) << b) {
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

template <xoshiro256_type X>
std::ostream& operator<<(std::ostream& os, const xoshiro256<X>& eng) {
    if (!(os << eng.state_[0] << ' ' << eng.state_[1] << ' ' << eng.state_[2] << ' ' << eng.state_[3])) {
        throw simplemc_exception("Error writing xoshiro256 to ostream.");
    }
    return os;
}

template <xoshiro256_type X>
std::istream& operator>>(std::istream& is, xoshiro256<X>& eng) {
    typename xoshiro256<X>::state_type arr;
    if (is >> arr[0] >> std::ws >> arr[1] >> std::ws >> arr[2] >> std::ws >> arr[3]) {
        eng.state_ = arr;
    } else {
        throw simplemc_exception("Error reading xoshiro256 from istream.");
    }
    return is;
}

template <xoshiro256_type X>
bool operator==(const xoshiro256<X>& lhs, const xoshiro256<X>& rhs) {
    return lhs.state_ == rhs.state_;
}

template <xoshiro256_type X>
bool operator!=(const xoshiro256<X>& lhs, const xoshiro256<X>& rhs) {
    return !(lhs == rhs);
}

/* Explicit template instantiations. */
template class xoshiro256<xoshiro256_type::plus>;
template class xoshiro256<xoshiro256_type::plusplus>;
template class xoshiro256<xoshiro256_type::starstar>;

template std::ostream& operator<<(std::ostream&, const xoshiro256p&);
template std::ostream& operator<<(std::ostream&, const xoshiro256pp&);
template std::ostream& operator<<(std::ostream&, const xoshiro256ss&);

template std::istream& operator>>(std::istream&, xoshiro256p&);
template std::istream& operator>>(std::istream&, xoshiro256pp&);
template std::istream& operator>>(std::istream&, xoshiro256ss&);

template bool operator==(const xoshiro256p&, const xoshiro256p&);
template bool operator==(const xoshiro256pp&, const xoshiro256pp&);
template bool operator==(const xoshiro256ss&, const xoshiro256ss&);

template bool operator!=(const xoshiro256p&, const xoshiro256p&);
template bool operator!=(const xoshiro256pp&, const xoshiro256pp&);
template bool operator!=(const xoshiro256ss&, const xoshiro256ss&);

} // namespace simplemc
