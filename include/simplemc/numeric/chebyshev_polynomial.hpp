/**
 * @file
 * @brief Recursive implementation of Chebyshev polynomials of the first and second kind.
 */

#ifndef SIMPLEMC_NUMERIC_CHEBYSHEV_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_CHEBYSHEV_POLYNOMIAL_HPP

#include <simplemc/numeric/orthogonal_polynomial.hpp>

#include <cassert>
#include <cmath>
#include <numbers>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-functions
 * @{
 */

/**
 * @brief Recursive implementation of Chebyshev polynomials of the first kind.
 *
 * @details Chebyshev polynomials of the first kind are defined on the interval \f$ [-1, 1] \f$. The
 * weight function is \f$ W(x) = 1 / \sqrt{1 - x^2} \f$ and it has an order dependent normalization
 * factor \f$ N_l \f$:
 * \f[
 *   \langle T_l, T_k \rangle = \int_{-1}^1 T_l(x) T_k(x) W(x) dx =
 *   \begin{cases}
 *   0 & \text{if } l \neq k \; , \\
 *   \pi & \text{if } l = k = 0 \; , \\
 *   \frac{\pi}{2} & \text{if } l = k \neq 0 \; .
 *   \end{cases}
 * \f]
 *
 * The recurrence relation is given by
 * \f[
 *   T_{l+1}(x) = 2x T_l(x) - T_{l-1}(x) \; ,
 * \f]
 * with \f$ T_0(x) = 1 \f$ and \f$ T_1(x) = x \f$.
 *
 * Then we get for the derivative
 * \f[
 *   \frac{d}{dx} T_{l+1}(x) = 2 \left(T_l(x) + x \frac{d}{dx} T_{l}(x) \right) - \frac{d}{dx}
 *   T_{l-1}(x) \; ,
 * \f]
 * with \f$ \frac{d}{dx} T_{0}(x) = 0 \f$ and \f$ \frac{d}{dx} T_{1}(x) = 1 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials) for more information.
 */
class chebyshev_polynomial_first : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Chebyshev polynomial of the first kind for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    chebyshev_polynomial_first(double x = 0.0);

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override;

    /**
     * @brief Get the value of the weight function evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override {
        assert(std::abs(x) <= 1.0);
        return 1.0 / std::sqrt(1 - x * x);
    }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @return Value of the l<sup>th</sup> order polynomial evaluated at the stored \f$ x \f$.
     */
    double next() override;
};

/**
 * @brief Recursive implementation of Chebyshev polynomials of the second kind.
 *
 * @details Chebyshev polynomials of the second kind are defined on the interval \f$ [-1, 1] \f$. The
 * weight function is \f$ W(x) = \sqrt{1 - x^2} \f$ and the order dependent normalization factor is
 * \f$ N_l = \pi / 2 \f$:
 * \f[
 *   \langle U_l, U_k \rangle = \int_{-1}^1 U_l(x) U_k(x) W(x) dx = \frac{\pi}{2} \delta_{lk} \; .
 * \f]
 *
 * The recurrence relation is given by
 * \f[
 *   U_{l+1}(x) = 2x U_l(x) - U_{l-1}(x) \; ,
 * \f]
 * with \f$ U_0(x) = 1 \f$ and \f$ U_1(x) = 2 x \f$.
 *
 * Then we get for the derivative
 * \f[
 *   \frac{d}{dx} U_{l+1}(x) = 2 \left(U_l(x) + x \frac{d}{dx} U_{l}(x) \right) - \frac{d}{dx}
 *   U_{l-1}(x) \; ,
 * \f]
 * with \f$ \frac{d}{dx} U_{0}(x) = 0 \f$ and \f$ \frac{d}{dx} U_{1}(x) = 2 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials) for more information.
 */
class chebyshev_polynomial_second : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Chebyshev polynomial of the second kind for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    chebyshev_polynomial_second(double x = 0.0);

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization([[maybe_unused]] int l) const override { return 2.0 * std::numbers::pi; }

    /**
     * @brief Get the value of the weight function evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override {
        assert(std::abs(x) <= 1.0);
        return std::sqrt(1 - x * x);
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

#endif // SIMPLEMC_NUMERIC_CHEBYSHEV_POLYNOMIAL_HPP
