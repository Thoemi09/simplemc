/**
 * @file
 * @brief Implementation details for simplemc/numeric/chebyshev_polynomial.hpp.
 */

#include <simplemc/numeric/chebyshev_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

chebyshev_polynomial_first::chebyshev_polynomial_first(double x) : x_(x), tlm1_(0), tl_(1), dtlm1_(0), dtl_(0) {
    if (std::abs(x_) > 1) {
        throw simplemc_exception("Chebyshev polynomials are only defined on the interval [-1, 1]",
            "chebyshev_polynomial_first::chebyshev_polynomial_first");
    }
}

double chebyshev_polynomial_first::norm(int l) const {
    assert(l >= 0);
    if (l == 0) {
        return std::numbers::pi;
    }
    return 0.5 * std::numbers::pi;
}

double chebyshev_polynomial_first::next() {
    double tlp1 = x_;
    double dtlp1 = 1;
    if (l_ > 0) {
        tlp1 = 2.0 * x_ * tl_ - tlm1_;
        dtlp1 = 2.0 * (tl_ + x_ * dtl_) - dtlm1_;
    }
    tlm1_ = tl_;
    tl_ = tlp1;
    dtlm1_ = dtl_;
    dtl_ = dtlp1;
    ++l_;
    return tlm1_;
}

chebyshev_polynomial_second::chebyshev_polynomial_second(double x) : x_(x), ulm1_(0), ul_(1), dulm1_(0), dul_(0) {
    if (std::abs(x_) > 1) {
        throw simplemc_exception("Chebyshev polynomials are only defined on the interval [-1, 1]",
            "chebyshev_polynomial_second::chebyshev_polynomial_second");
    }
}

double chebyshev_polynomial_second::next() {
    double ulp1 = 2.0 * x_;
    double dulp1 = 2.0;
    if (l_ > 0) {
        ulp1 = 2.0 * x_ * ul_ - ulm1_;
        dulp1 = 2.0 * (ul_ + x_ * dul_) - dulm1_;
    }
    ulm1_ = ul_;
    ul_ = ulp1;
    dulm1_ = dul_;
    dul_ = dulp1;
    ++l_;
    return ulm1_;
}

} // namespace simplemc
