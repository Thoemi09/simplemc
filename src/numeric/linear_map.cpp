/**
 * @file linear_map.cpp
 * @brief Linear map from a range [a, b] to another range [c, d] (and vice versa).
 */

#include <simplemc/numeric/linear_map.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cassert>
#include <limits>

namespace simplemc {

linear_map::linear_map(const range_type& range1, const range_type& range2) {
    set(range1, range2);
}

void linear_map::set(const range_type& range1, const range_type& range2) {
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto minus_inf = -inf;
    double a = range1[0];
    double b = range1[1];
    double c = range2[0];
    double d = range2[1];
    if (a >= b || c >= d) {
        throw simplemc_exception("Wrong intervals in linear map", "linear_map::set");
    }
    if (a > minus_inf && b < inf && c > minus_inf && d < inf) {
        alpha_ = (c - d) / (a - b);
        beta_ = c - a * alpha_;
    } else if (a == minus_inf && b != inf && c == minus_inf && d != inf) {
        alpha_ = 1.0;
        beta_ = d - b;
    } else if (a != minus_inf && b == inf && c != minus_inf && d == inf) {
        alpha_ = 1.0;
        beta_ = c - a;
    } else if (a == minus_inf && b == inf && c == minus_inf && d == inf) {
        alpha_ = 1.0;
        beta_ = 0.0;
    } else {
        throw simplemc_exception("Wrong intervals in linear map", "linear_map::set");
    }
}

double linear_map::map(double x) const {
    assert(x >= range1_[0] && x <= range1_[1]);
    return alpha_ * x + beta_;
}

double linear_map::inverse_map(double y) const {
    assert(y >= range2_[0] && y <= range2_[1]);
    return (y - beta_) / alpha_;
}

} // namespace simplemc