/**
 * @file
 * @brief Recursive implementation of Hermite polynomials.
 */

#ifndef SIMPLEMC_NUMERIC_HERMITE_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_HERMITE_POLYNOMIAL_HPP

#include <simplemc/numeric/orthogonal_polynomial.hpp>

#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-functions
 * @{
 */

/**
 * @brief Recursive implementation of Hermite polynomials.
 *
 * @details Hermite polynomials are defined on the interval \f$ (-\infty, \infty) \f$. The weight
 * function is \f$ W(x) = e^{-x^2} \f$ and the order depdendent normalization factor is \f$ N_l =
 * \sqrt{\pi} 2^l l! \f$:
 * \f[
 *   \langle H_l, H_k \rangle = \int_{-\infty}^\infty H_l(x) H_k(x) W(x) dx = \sqrt{\pi} 2^l l!
 *   \delta_{lk} \; .
 * \f]
 *
 * The recurrence relations is given by
 * \f[
 *   H_{l+1}(x) = 2 x H_l(x) - 2 l H_{l-1}(x) \; ,
 * \f]
 * with with \f$ H_0(x) = 1 \f$ and \f$ H_1(x) = 2x \f$.
 *
 * For the derivative we have
 * \f[
 *   \frac{d}{dx} H_{l+1}(x) = 2 (l + 1) H_l(x) \; ,
 * \f]
 * with \f$ \frac{d}{dx} H_{0}(x) = 0 \f$ and \f$ \frac{d}{dx} H_{1}(x) = 2 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Hermite_polynomials) for more information.
 */
class hermite_polynomial : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Hermite polynomial for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    hermite_polynomial(double x = 0.0);

    /**
     * @brief Get the domain on which the polynomial is defined.
     *
     * @return `std::array<double, 2>` representing the domain.
     */
    [[nodiscard]] std::array<double, 2> domain() const override {
        return { -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
    }

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization([[maybe_unused]] int l) const override {
        assert(l >= 0);
        return std::pow(2, l) * std::tgamma(l + 1) / std::numbers::inv_sqrtpi;
    }

    /**
     * @brief Get the value of the weight function evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override { return std::exp(-x * x); }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @return Value of the l<sup>th</sup> order polynomial evaluated at the stored \f$ x \f$.
     */
    double next() override;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_HERMITE_POLYNOMIAL_HPP
