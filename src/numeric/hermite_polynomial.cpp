/**
 * @file
 * @brief Implementation details for simplemc/numeric/hermite_polynomial.hpp.
 */

#include <simplemc/numeric/hermite_polynomial.hpp>

namespace simplemc {

hermite_polynomial::hermite_polynomial(double x) : orthogonal_polynomial { x, 1.0, 2.0 * x, 0.0, 2.0 } {}

double hermite_polynomial::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            dyl_ = dy1_;
            dylm1_ = dy0_;
            break;
        default:
            double ylp1 = 2.0 * x_ * yl_ - 2.0 * l_ * ylm1_;
            double dylp1 = 2.0 * (l_ + 1) * yl_;
            ylm1_ = yl_;
            yl_ = ylp1;
            dylm1_ = dyl_;
            dyl_ = dylp1;
    }
    ++l_;
    return ylm1_;
}

} // namespace simplemc
