/**
 * @file
 * @brief Implementation details for simplemc/numeric/utils.hpp.
 */

#include <simplemc/numeric/utils.hpp>

#include <cassert>

namespace simplemc {

double map_to_interval(double val, double lower_bound, double upper_bound) {
    assert(upper_bound > lower_bound);
    const auto len = upper_bound - lower_bound;
    const auto mid = (upper_bound + lower_bound) * 0.5;
    const auto res =
        val > mid ? std::fmod(val - lower_bound, len) + lower_bound : std::fmod(val - upper_bound, len) + upper_bound;
    return (res == lower_bound ? upper_bound : res);
}

double map_to_interval_lb(double val, double lower_bound, double upper_bound) {
    assert(upper_bound > lower_bound);
    const auto len = upper_bound - lower_bound;
    const auto mid = (upper_bound + lower_bound) * 0.5;
    const auto res =
        val > mid ? std::fmod(val - lower_bound, len) + lower_bound : std::fmod(val - upper_bound, len) + upper_bound;
    return (res == upper_bound ? lower_bound : res);
}

} // namespace simplemc
