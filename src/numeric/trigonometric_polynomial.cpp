/**
 * @file
 * @brief Implementation details for simplemc/numeric/trigonometric_polynmoial.hpp.
 */

#include <simplemc/numeric/trigonometric_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>

namespace simplemc {

cosine::cosine(double x) :
    orthogonal_polynomial { x, 1.0, std::cos(x), 0.0, 1.0 },
    cosx_ { y1_ },
    sinx_ { std::sin(x) } {}

double cosine::normalization(int l) const {
    assert(l >= 0);
    if (l == 0) {
        return 2.0 * std::numbers::pi;
    }
    return std::numbers::pi;
}

double cosine::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            dyl_ = dy1_;
            dylm1_ = dy0_;
            break;
        default:
            double ylp1 = 2.0 * cosx_ * yl_ - ylm1_;
            double dylp1 = 2.0 * cosx_ * dyl_ - dylm1_;
            ylm1_ = yl_;
            yl_ = ylp1;
            dylm1_ = dyl_;
            dyl_ = dylp1;
    }
    ++l_;
    return ylm1_;
}

sine::sine(double x) : orthogonal_polynomial { x, 0.0, 1.0, 1.0, std::cos(x) }, cosx_ { dy1_ }, sinx_ { std::sin(x) } {}

double sine::normalization(int l) const {
    assert(l >= 0);
    if (l == 0) {
        return 0.0;
    }
    return std::numbers::pi;
}

double sine::next() {
    switch (l_) {
        case 0:
            yl_ = y1_;
            ylm1_ = y0_;
            dyl_ = dy1_;
            dylm1_ = dy0_;
            break;
        default:
            double ylp1 = 2.0 * cosx_ * yl_ - ylm1_;
            double dylp1 = 2.0 * cosx_ * dyl_ - dylm1_;
            ylm1_ = yl_;
            yl_ = ylp1;
            dylm1_ = dyl_;
            dyl_ = dylp1;
    }
    ++l_;
    return ylm1_ * sinx_;
}

} // namespace simplemc
