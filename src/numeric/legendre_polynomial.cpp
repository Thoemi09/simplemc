/**
 * @file
 * @brief Implementation details for simplemc/numeric/legendre_polynomial.hpp.
 */

#include <simplemc/numeric/legendre_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>

namespace simplemc {

legendre_polynomial::legendre_polynomial(double x) : orthogonal_polynomial { x, 1.0, x, 0.0, 1.0 }, ylm2_ { 0.0 } {
    if (std::abs(x_) > 1) {
        throw simplemc_exception("Legendre polynomials are only defined on the interval [-1, 1]",
            "legendre_polynomial::legendre_polynomial");
    }
}

double legendre_polynomial::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            break;
        default:
            ylm2_ = ylm1_;
            ylm1_ = yl_;
            yl_ = ((2 * l_ + 1) * x_ * ylm1_ - l_ * ylm2_) / (l_ + 1);
    }
    ++l_;
    return ylm1_;
}

double legendre_polynomial::current_derivative() const {
    switch (l_) { // NOLINT (no default is intended)
        case 0: return 0;
        case 1: return 1;
    }
    if (std::abs(x_) == 1) {
        return l_ * (l_ + 1) * 0.5 * std::pow(-1.0, l_ + 1);
    }
    return l_ / (x_ * x_ - 1) * (x_ * yl_ - ylm1_);
}

double legendre_polynomial::last_derivative() const {
    assert(l_ > 0);
    switch (l_) { // NOLINT (no default is intended)
        case 1: return 0;
        case 2: return 1;
    }
    if (std::abs(x_) == 1) {
        return (l_ - 1) * l_ * 0.5 * std::pow(-1.0, l_);
    }
    return (l_ - 1) / (x_ * x_ - 1) * (x_ * ylm1_ - ylm2_);
}

} // namespace simplemc
