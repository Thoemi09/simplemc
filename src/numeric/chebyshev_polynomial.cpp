/**
 * @file
 * @brief Implementation details for simplemc/numeric/chebyshev_polynomial.hpp.
 */

#include <simplemc/numeric/chebyshev_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

chebyshev_polynomial_first::chebyshev_polynomial_first(double x) : orthogonal_polynomial { x, 1.0, x, 0.0, 1.0 } {
    if (std::abs(x_) > 1) {
        throw simplemc_exception("Chebyshev polynomials are only defined on the interval [-1, 1]",
            "chebyshev_polynomial_first::chebyshev_polynomial_first");
    }
}

double chebyshev_polynomial_first::normalization(int l) const {
    assert(l >= 0);
    switch (l) { // NOLINT (no default is intended)
        case 0: return std::numbers::pi;
    }
    return 2.0 * std::numbers::pi;
}

double chebyshev_polynomial_first::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            dyl_ = dy1_;
            dylm1_ = dy0_;
            break;
        default:
            double ylp1 = 2.0 * x_ * yl_ - ylm1_;
            double dylp1 = 2.0 * (yl_ + x_ * dyl_) - dylm1_;
            ylm1_ = yl_;
            yl_ = ylp1;
            dylm1_ = dyl_;
            dyl_ = dylp1;
    }
    ++l_;
    return ylm1_;
}

chebyshev_polynomial_second::chebyshev_polynomial_second(double x) :
    orthogonal_polynomial { x, 1.0, 2.0 * x, 0.0, 2.0 } {
    if (std::abs(x_) > 1) {
        throw simplemc_exception("Chebyshev polynomials are only defined on the interval [-1, 1]",
            "chebyshev_polynomial_second::chebyshev_polynomial_second");
    }
}

double chebyshev_polynomial_second::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            dyl_ = dy1_;
            dylm1_ = dy0_;
            break;
        default:
            double ylp1 = 2.0 * x_ * yl_ - ylm1_;
            double dylp1 = 2.0 * (yl_ + x_ * dyl_) - dylm1_;
            ylm1_ = yl_;
            yl_ = ylp1;
            dylm1_ = dyl_;
            dyl_ = dylp1;
    }
    ++l_;
    return ylm1_;
}

} // namespace simplemc
