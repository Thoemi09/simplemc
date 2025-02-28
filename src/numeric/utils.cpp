/**
 * @file
 * @brief Implementation details for simplemc/numeric/utils.hpp.
 */

#include <simplemc/numeric/utils.hpp>

#include <cassert>
#include <cmath>

namespace simplemc {

double map_to_interval(double x, double a, double b) {
    assert(b > a);
    const auto len = b - a;
    const auto mid = (b + a) * 0.5;
    const auto res = x > mid ? std::fmod(x - a, len) + a : std::fmod(x - b, len) + b;
    return (res == a ? b : res);
}

double map_to_interval_lb(double x, double a, double b) {
    assert(b > a);
    const auto len = b - a;
    const auto mid = (b + a) * 0.5;
    const auto res = x > mid ? std::fmod(x - a, len) + a : std::fmod(x - b, len) + b;
    return (res == b ? a : res);
}

} // namespace simplemc
