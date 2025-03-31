/**
 * @file
 * @brief Implementation details for simplemc/numeric/legendre_polynomial.hpp.
 */

#include <simplemc/numeric/legendre_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>

namespace simplemc {

legendre_polynomial::legendre_polynomial(double x) : x_(x), plm1_(0), pl_(1), dplm1_(0), dpl_(0) {
    if (std::abs(x_) > 1) {
        throw simplemc_exception("Legendre polynomials are only defined on the interval [-1, 1]",
            "legendre_polynomial::legendre_polynomial");
    }
}

double legendre_polynomial::next() {
    double plp1 = x_;
    double dplp1 = 1;
    if (l_ > 0) {
        plp1 = ((2 * l_ + 1) * x_ * pl_ - l_ * plm1_) / (l_ + 1);
        dplp1 = (l_ + 1) * pl_ + x_ * dpl_;
    }
    plm1_ = pl_;
    pl_ = plp1;
    dplm1_ = dpl_;
    dpl_ = dplp1;
    ++l_;
    return plm1_;
}

} // namespace simplemc
