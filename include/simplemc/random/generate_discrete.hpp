/**
 * @file generate_discrete.hpp
 * @brief Generate an int from a discrete distribution.
 */

#ifndef SIMPLEMC_RANDOM_GENERATE_DISCRETE_HPP
#define SIMPLEMC_RANDOM_GENERATE_DISCRETE_HPP

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace simplemc::detail {

/**
 * @brief Generate a discrete random int using a linear search.
 *
 * @tparam IntType Integer type to be returned.
 * @tparam Engine RNG.
 * @param eng RNG.
 * @param vec Vector of accumulated probabilities in increasing order.
 * @return First integer i such that vec[i] >= uni01(eng).
 */
template <typename IntType, typename Engine>
inline IntType generate_discrete_ls(Engine& eng, const std::vector<double>& vec) {
    assert(vec.size() != 0);
    const auto r = std::uniform_real_distribution<double> { 0.0, 1.0 }(eng);
    const auto size = static_cast<IntType>(vec.size());
    for (IntType i = 0; i < size; ++i) {
        if (vec[i] >= r)
            return i;
    }
    return size - 1;
}

/**
 * @brief Generate a discrete random int using std::lower_bound.
 *
 * @tparam IntType Integer type to be returned.
 * @tparam Engine RNG.
 * @param eng RNG.
 * @param vec Vector of accumulated probabilities in increasing order.
 * @return First integer i such that vec[i] >= uni01(eng).
 */
template <typename IntType, typename Engine>
inline IntType generate_discrete_lb(Engine& eng, const std::vector<double>& vec) {
    const auto r = std::uniform_real_distribution<double> { 0.0, 1.0 }(eng);
    auto pos = std::lower_bound(vec.begin(), vec.end(), r);
    return static_cast<IntType>(pos - vec.begin());
}

} // namespace simplemc::detail

#endif // SIMPLEMC_RANDOM_GENERATE_DISCRETE_HPP
