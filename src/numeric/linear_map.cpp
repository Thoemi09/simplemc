/**
 * @file
 * @brief Implementation details for simplemc/numeric/linear_map.hpp.
 */

#include <simplemc/numeric/linear_map.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <limits>

namespace simplemc {

void linear_map::set(const interval_type& dom, const interval_type& rg) {
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto minus_inf = -inf;
    double a = dom[0];
    double b = dom[1];
    double c = rg[0];
    double d = rg[1];
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
    domain_ = dom;
    range_ = rg;
}

} // namespace simplemc
