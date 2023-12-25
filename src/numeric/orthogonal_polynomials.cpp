/**
 * @file special_functions.cpp
 * @brief Implementation of special functions for estimating coefficients in a generalized Fourier series.
 */

#include <simplemc/numeric/orthogonal_polynomials.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

orthogonal_polynomial::orthogonal_polynomial(double x, double y0, double y1, double dy0, double dy1) :
    l_ { 0 },
    x_ { x },
    y0_ { y0 },
    y1_ { y1 },
    yl_ { y0 },
    ylm1_ { 0.0 },
    dy0_ { dy0 },
    dy1_ { dy1 },
    dyl_ { dy0 },
    dylm1_ { 0.0 } {}

double orthogonal_polynomial::last_value() const {
    assert(l_ > 0);
    return ylm1_;
}

double orthogonal_polynomial::last_derivative() const {
    assert(l_ > 0);
    return dylm1_;
}

double orthogonal_polynomial::derivative(int l, double x) {
    assert(l >= 0);
    assert(x >= lower_bound() && x <= upper_bound());
    value(l, x);
    return current_derivative();
}

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

double legendre_polynomial::value(int l, double x) {
    assert(l >= 0);
    assert(x >= lower_bound() && x <= upper_bound());
    l_ = 0;
    x_ = x;
    y0_ = 1.0;
    y1_ = x_;
    yl_ = y0_;
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_;
}

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

double chebyshev_polynomial_first::value(int l, double x) {
    assert(l >= 0);
    assert(x >= lower_bound() && x <= upper_bound());
    l_ = 0;
    x_ = x;
    y0_ = 1.0;
    y1_ = x_;
    yl_ = y0_;
    dy0_ = 0.0;
    dy1_ = 1.0;
    dyl_ = dy0_;
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_;
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

double chebyshev_polynomial_second::value(int l, double x) {
    assert(l >= 0);
    assert(x >= lower_bound() && x <= upper_bound());
    l_ = 0;
    x_ = x;
    y0_ = 1.0;
    y1_ = 2.0 * x_;
    yl_ = y0_;
    dy0_ = 0.0;
    dy1_ = 2.0;
    dyl_ = dy0_;
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_;
}

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

double laguerre_polynomial::value(int l, double x) {
    assert(l >= 0);
    assert(x >= lower_bound() && x <= upper_bound());
    l_ = 0;
    x_ = x;
    y0_ = 1.0;
    y1_ = 1.0 - x_;
    yl_ = y0_;
    dy0_ = 0.0;
    dy1_ = -1.0;
    dyl_ = dy0_;
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_;
}

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

double hermite_polynomial::value(int l, double x) {
    assert(l >= 0);
    assert(x >= lower_bound() && x <= upper_bound());
    l_ = 0;
    x_ = x;
    y0_ = 1.0;
    y1_ = 2.0 * x_;
    yl_ = y0_;
    dy0_ = 0.0;
    dy1_ = 2.0;
    dyl_ = dy0_;
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_;
}

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

double cosine::current_derivative() const {
    return -static_cast<double>(l_) * dyl_ * sinx_;
}

double cosine::last_derivative() const {
    assert(l_ > 0);
    return -static_cast<double>(l_ - 1) * dylm1_ * sinx_;
}

double cosine::value(int l, double x) {
    assert(l >= 0);
    l_ = 0;
    x_ = x;
    y0_ = 1.0;
    y1_ = std::cos(x_);
    yl_ = y0_;
    dy0_ = 0.0;
    dy1_ = 1.0;
    dyl_ = dy0_;
    cosx_ = y1_;
    sinx_ = std::sin(x);
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_;
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

double sine::current_value() const {
    return yl_ * sinx_;
}

double sine::last_value() const {
    assert(l_ > 0);
    return ylm1_ * sinx_;
}

double sine::current_derivative() const {
    return static_cast<double>(l_) * dyl_;
}

double sine::last_derivative() const {
    assert(l_ > 0);
    return static_cast<double>(l_ - 1) * dylm1_;
}

double sine::value(int l, double x) {
    assert(l >= 0);
    l_ = 0;
    x_ = x;
    y0_ = 0.0;
    y1_ = 1.0;
    yl_ = y0_;
    dy0_ = 1.0;
    dy1_ = std::cos(x);
    dyl_ = dy0_;
    cosx_ = dy1_;
    sinx_ = std::sin(x);
    for (int i = 0; i < l; ++i) {
        next();
    }
    return yl_ * sinx_;
}

} // namespace simplemc
