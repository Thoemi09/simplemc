/**
 * @file
 * @brief Implementation details for simplemc/numeric/hermite_polynomial.hpp.
 */

#include <simplemc/numeric/hermite_polynomial.hpp>

namespace simplemc {

hermite_polynomial::hermite_polynomial(double x) : x_(x), hlm1_(0), hl_(1), dhlm1_(0), dhl_(0) {}

double hermite_polynomial::next() {
    double hlp1 = 2.0 * x_;
    double dhp1 = 2.0;
    if (l_ > 0) {
        hlp1 = 2.0 * x_ * hl_ - 2.0 * l_ * hlm1_;
        dhp1 = 2.0 * (l_ + 1) * hl_;
    }
    hlm1_ = hl_;
    hl_ = hlp1;
    dhlm1_ = dhl_;
    dhl_ = dhp1;
    ++l_;
    return hlm1_;
}

} // namespace simplemc
