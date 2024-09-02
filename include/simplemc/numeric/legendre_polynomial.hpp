/**
 * @file
 * @brief Recursive implementation of Legendre polynomials.
 */

#ifndef SIMPLEMC_NUMERIC_LEGENDRE_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_LEGENDRE_POLYNOMIAL_HPP

#include <simplemc/numeric/orthogonal_polynomial.hpp>

#include <cassert>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-functions
 * @{
 */

/**
 * @brief Recursive implementation of Legendre polynomials.
 *
 * @details Legendre polynomials are defined on the interval \f$ [-1, 1] \f$. The weight function is
 * \f$ W(x) = 1 \f$ and the order dependent normalization factor is \f$ N_l = 2 / (2l + 1) \f$:
 * \f[
 *   \langle p_l, p_k \rangle = \int_{-1}^1 p_l(x) p_k(x) dx = \frac{2}{2l + 1} \delta_{lk} \; .
 * \f]
 *
 * The recurrence relation is given by
 * \f[
 *   p_{l+1}(x) = \frac{(2l + 1) x p_l(x) - l p_{l-1}(x)}{l + 1} \; ,
 * \f]
 * with with \f$ p_0(x) = 1 \f$ and \f$ p_1(x) = x \f$.
 *
 * For the derivative we have
 * \f[
 *   \frac{d}{dx} p_{l+1}(x) = (l + 1) p_l(x) + x \frac{d}{dx} p_{l}(x) \; ,
 * \f]
 * with \f$ \frac{d}{dx} p_{0}(x) = 0 \f$ and \f$ \frac{d}{dx} p_{1}(x) = 1 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Legendre_polynomials) for more information.
 */
class legendre_polynomial : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Legendre polynomial for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    legendre_polynomial(double x = 0.0);

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override {
        assert(l >= 0);
        return 2.0 / (2.0 * l + 1);
    }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @return Value of the l<sup>th</sup> order polynomial evaluated at the stored \f$ x \f$.
     */
    double next() override;

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double current_derivative() const override;

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double last_derivative() const override;

private:
    double ylm2_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LEGENDRE_POLYNOMIAL_HPP
