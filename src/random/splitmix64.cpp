/**
 * @file splitmix64.cpp
 * @brief Random number generator splitmix64.
 */

#include <simplemc/random/splitmix64.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <iostream>

namespace simplemc {

splitmix64::splitmix64(std::uint64_t s) : state_(s) {}

void splitmix64::discard(std::uintmax_t z) {
    for (std::uintmax_t i = 0; i < z; i++) {
        this->operator()();
    }
}

std::ostream& operator<<(std::ostream& os, const splitmix64& eng) {
    if (!(os << eng.state_)) {
        throw simplemc_exception("Error writing splitmix64 to ostream");
    }
    return os;
}

std::istream& operator>>(std::istream& is, splitmix64& eng) {
    std::uint64_t s {};
    if (is >> s) {
        eng.state_ = s;
    } else {
        throw simplemc_exception("Error reading splitmix64 from istream");
    }
    return is;
}

bool operator==(const splitmix64& lhs, const splitmix64& rhs) {
    return lhs.state_ == rhs.state_;
}

bool operator!=(const splitmix64& lhs, const splitmix64& rhs) {
    return !(lhs == rhs);
}

} // namespace simplemc
