/**
 * @file
 * @brief Recursive implementation of Laguerre polynomials.
 */

#ifndef SIMPLEMC_NUMERIC_LAGUERRE_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_LAGUERRE_POLYNOMIAL_HPP

#include <simplemc/numeric/orthogonal_polynomial.hpp>

#include <cassert>
#include <cmath>
#include <limits>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-functions
 * @{
 */

/**
 * @brief Recursive implementation of Laguerre polynomials.
 *
 * @details Laguerre polynomials are defined on the interval \f$ [0, \infty) \f$. The weight function
 * is \f$ W(x) = e^{-x} \f$ and the constant normalization factor is \f$ N_l = 1 \f$:
 * \f[
 *   \langle L_l, L_k \rangle = \int_0^\infty L_l(x) L_k(x) W(x) dx = \delta_{lk} \; .
 * \f]
 *
 * The recurrence relations is given by
 * \f[
 *   L_{l+1}(x) = \frac{(2l + 1 - x) L_l(x) - l L_{l-1}(x)}{l + 1} \; ,
 * \f]
 * with with \f$ L_0(x) = 1 \f$ and \f$ L_1(x) = 1 - x \f$.
 *
 * Then we get for the derivative
 * \f[
 *   \frac{d}{dx} L_{l+1}(x) = \frac{(2l + 1 - x) \frac{d}{dx} L_l(x) - L_l(x) - l \frac{d}{dx}
 *   L_{l-1}(x)}{l + 1} \; ,
 * \f]
 * with \f$ \frac{d}{dx} L_{0}(x) = 0 \f$ and \f$ \frac{d}{dx} L_{1}(x) = -1 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Laguerre_polynomials) for more information.
 */
class laguerre_polynomial : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Laguerre polynomial for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    laguerre_polynomial(double x = 0.0);

    /**
     * @brief Get the domain on which the polynomial is defined.
     *
     * @return `std::array<double, 2>` representing the domain.
     */
    [[nodiscard]] std::array<double, 2> domain() const override {
        return { 0.0, std::numeric_limits<double>::infinity() };
    }

    /**
     * @brief Get the value of the weight function evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override {
        assert(x >= 0.0);
        return std::exp(-x);
    }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @return Value of the l<sup>th</sup> order polynomial evaluated at the stored \f$ x \f$.
     */
    double next() override;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LAGUERRE_POLYNOMIAL_HPP
