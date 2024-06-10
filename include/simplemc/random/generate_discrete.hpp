/**
 * @file
 * @brief Generate an integer from a discrete distribution.
 */

#ifndef SIMPLEMC_RANDOM_GENERATE_DISCRETE_HPP
#define SIMPLEMC_RANDOM_GENERATE_DISCRETE_HPP

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace simplemc::detail {

// Generate a discrete random integer using a linear search on a vector of accumulated probabilities.
template <typename IntType, typename RNG>
inline IntType generate_discrete_ls(RNG& rng, const std::vector<double>& vec) {
    assert(vec.size() != 0);
    const auto r = std::uniform_real_distribution<double> { 0.0, 1.0 }(rng);
    const auto size = static_cast<IntType>(vec.size());
    for (IntType i = 0; i < size; ++i) {
        if (vec[i] >= r) {
            return i;
        }
    }
    return size - 1;
}

// Generate a discrete random integer using std::lower_bound on a vector of accumulated probabilities.
template <typename IntType, typename RNG>
inline IntType generate_discrete_lb(RNG& rng, const std::vector<double>& vec) {
    assert(vec.size() != 0);
    const auto r = std::uniform_real_distribution<double> { 0.0, 1.0 }(rng);
    auto pos = std::lower_bound(vec.begin(), vec.end(), r);
    return static_cast<IntType>(pos - vec.begin());
}

} // namespace simplemc::detail

#endif // SIMPLEMC_RANDOM_GENERATE_DISCRETE_HPP
