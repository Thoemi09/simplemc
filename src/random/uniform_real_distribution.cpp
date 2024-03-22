/**
 * @file
 * @brief Uniform distribution on the real line.
 */

#include <simplemc/random/uniform_real_distribution.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cassert>
#include <iostream>
#include <limits>

namespace simplemc {

namespace detail {

urd_param_type::urd_param_type(double min_arg, double max_arg) : min_(min_arg), max_(max_arg) {
    assert(min_ < max_);
}

std::ostream& operator<<(std::ostream& os, const urd_param_type& param) {
    auto prec = os.precision();
    os.precision(std::numeric_limits<double>::max_digits10);
    if (!(os << param.min() << ' ' << param.max())) {
        throw simplemc_exception("Error writing urd_param_type to ostream");
    }
    os.precision(prec);
    return os;
}

std::istream& operator>>(std::istream& is, urd_param_type& param) {
    double min = 0, max = 0;
    if (is >> min >> std::ws >> max) {
        if (min < max) {
            param = urd_param_type { min, max };
        } else {
            is.setstate(std::ios_base::failbit);
        }
    } else {
        throw simplemc_exception("Error reading urd_param_type from istream");
    }
    return is;
}

bool operator==(const urd_param_type& lhs, const urd_param_type& rhs) {
    return (lhs.min() == rhs.min() && lhs.max() == rhs.max());
}

bool operator!=(const urd_param_type& lhs, const urd_param_type& rhs) {
    return !(lhs == rhs);
}

} // namespace detail

uniform_real_distribution::uniform_real_distribution(double min_arg, double max_arg) : param_(min_arg, max_arg) {}

uniform_real_distribution::uniform_real_distribution(const param_type& param) : param_(param) {}

std::ostream& operator<<(std::ostream& os, const uniform_real_distribution& ud) {
    if (!(os << ud.param())) {
        throw simplemc_exception("Error writing uniform_real_distribution to ostream");
    }
    return os;
}

std::istream& operator>>(std::istream& is, uniform_real_distribution& ud) {
    uniform_real_distribution::param_type parm;
    if (is >> parm) {
        ud.param(parm);
    } else {
        throw simplemc_exception("Error reading uniform_real_distribution from istream");
    }
    return is;
}

bool operator==(const uniform_real_distribution& lhs, const uniform_real_distribution& rhs) {
    return (lhs.param() == rhs.param());
}

bool operator!=(const uniform_real_distribution& lhs, const uniform_real_distribution& rhs) {
    return !(lhs == rhs);
}

} // namespace simplemc
