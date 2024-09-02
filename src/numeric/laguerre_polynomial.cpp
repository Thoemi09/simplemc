/**
 * @file
 * @brief Implementation details for simplemc/numeric/laguerre_polynomial.cpp.
 */

#include <simplemc/numeric/laguerre_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

laguerre_polynomial::laguerre_polynomial(double x) : orthogonal_polynomial { x, 1.0, 1.0 - x, 0.0, -1.0 } {
    if (x < 0) {
        throw simplemc_exception("Lagurre polynomials are only defined on the interval [0, inf)",
            "laguerre_polynomial::laguerre_polynomial");
    }
}

double laguerre_polynomial::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            dyl_ = dy1_;
            dylm1_ = dy0_;
            break;
        default:
            double a = -1.0 / (l_ + 1);
            double b = 2.0 + a;
            double c = 1.0 + a;
            double ylp1 = (a * x_ + b) * yl_ - c * ylm1_;
            double dylp1 = a * yl_ + (a * x_ + b) * dyl_ - c * dylm1_;
            ylm1_ = yl_;
            yl_ = ylp1;
            dylm1_ = dyl_;
            dyl_ = dylp1;
    }
    ++l_;
    return ylm1_;
}

} // namespace simplemc
