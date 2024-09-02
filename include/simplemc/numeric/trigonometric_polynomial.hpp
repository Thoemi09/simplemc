/**
 * @file
 * @brief Recursive implementation of trigonometric polynomials.
 */

#ifndef SIMPLEMC_NUMERIC_TRIGONOMETRIC_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_TRIGONOMETRIC_POLYNOMIAL_HPP

#include <simplemc/numeric/orthogonal_polynomial.hpp>

#include <array>
#include <cassert>
#include <numbers>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-functions
 * @{
 */

/**
 * @brief Recursive implementation of \f$ \cos(l x) \f$.
 *
 * @details \f$ C_l(x) = \cos(l x) \f$ is defined on the interval \f$ [-\pi, \pi] \f$. The weight
 * function is \f$ W(x) = 1 \f$ and it has an order dependent normalization factor \f$ N_l \f$:
 * \f[
 *   \langle C_l, C_k \rangle = \int_{-\pi}^\pi C_l(x) C_k(x) dx =
 *   \begin{cases}
 *   0 & \text{if } l \neq k \; , \\
 *   2 \pi & \text{if } l = k = 0 \; , \\
 *   \pi & \text{if } l = k \neq 0 \; .
 *   \end{cases}
 * \f]
 *
 * The recurrence relation is given in terms of Chebyshev polynomials of the first kind
 * \f[
 *   C_{l+1}(x) = T_{l+1}(\cos(x)) \; ,
 * \f]
 * with \f$ C_0(x) = 1 \f$, \f$ C_1(x) = \cos(x) \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials#Trigonometric_definition) and
 * simplemc::chebyshev_polynomial_first for more information.
 */
class cosine : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a cosine for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    cosine(double x = 0.0);

    /**
     * @brief Get the domain on which the polynomial is defined.
     *
     * @return `std::array<double, 2>` representing the domain.
     */
    [[nodiscard]] std::array<double, 2> domain() const override { return { -std::numbers::pi, std::numbers::pi }; }

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override;

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
    [[nodiscard]] double current_derivative() const override { return -static_cast<double>(l_) * dyl_ * sinx_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double last_derivative() const override {
        assert(l_ > 0);
        return -static_cast<double>(l_ - 1) * dylm1_ * sinx_;
    }

private:
    double cosx_;
    double sinx_;
};

/**
 * @brief Recursive implementation of \f$ \sin(l x) \f$.
 *
 * @details \f$ S_l(x) = \sin(l x) \f$ is defined on the interval \f$ [-\pi, \pi] \f$. The weight
 * function is \f$ W(x) = 1 \f$ and it has an order dependent normalization factor \f$ N_l \f$:
 * \f[
 *   \langle S_l, S_k \rangle = \int_{-\pi}^\pi S_l(x) S_k(x) dx =
 *   \begin{cases}
 *   0 & \text{if } l \neq k \; , \\
 *   0 \pi & \text{if } l = k = 0 \; , \\
 *   \pi & \text{if } l = k \neq 0 \; .
 *   \end{cases}
 * \f]
 *
 * The recurrence relation is given in terms of Chebyshev polynomials of the second kind
 * \f[
 *   S_{l+1}(x) = U_l(\cos(x)) \sin(x) \; ,
 * \f]
 * with \f$ S_0(x) = 0 \f$, \f$ S_1(x) = \sin(x) \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials#Trigonometric_definition) and
 * simplemc::chebyshev_polynomial_second for more information.
 */
class sine : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a sine for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomial is evaluated.
     */
    sine(double x = 0.0);

    /**
     * @brief Get the domain on which the polynomial is defined.
     *
     * @return `std::array<double, 2>` representing the domain.
     */
    [[nodiscard]] std::array<double, 2> domain() const override { return { -std::numbers::pi, std::numbers::pi }; }

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override;

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @return Value of the l<sup>th</sup> order polynomial evaluated at the stored \f$ x \f$.
     */
    double next() override;

    /**
     * @brief Get the value of the currently processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double current_value() const override { return yl_ * sinx_; }

    /**
     * @brief Get the value of the previously processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double last_value() const override {
        assert(l_ > 0);
        return ylm1_ * sinx_;
    }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double current_derivative() const override { return static_cast<double>(l_) * dyl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] double last_derivative() const override {
        assert(l_ > 0);
        return static_cast<double>(l_ - 1) * dylm1_;
    }

private:
    double cosx_;
    double sinx_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_TRIGONOMETRIC_POLYNOMIAL_HPP
