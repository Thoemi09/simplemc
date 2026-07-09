/**
 * @file
 * @brief Implementation of Hermite polynomials using recurrence relations.
 */

#ifndef SIMPLEMC_NUMERIC_HERMITE_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_HERMITE_POLYNOMIAL_HPP

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-poly
 * @{
 */

/**
 * @brief Implementation of Hermite polynomials using recurrence relations.
 *
 * @details In the following, we use the notation from @ref simplemc-numeric-poly.
 *
 * Hermite polynomials, \f$ H_l(x) \f$, are defined on the interval \f$ \mathrm{D} = (-\infty, \infty)
 * \f$. Their weight function is \f$ W(x) = e^{-x^2} \f$ and their order dependent normalization
 * factor is \f$ N_l = \sqrt{\pi} 2^l l! \f$:
 * \f[
 *   \langle H_l, H_k \rangle = \int_{-\infty}^\infty H_l(x) H_k(x) e^{-x^2} dx = \sqrt{\pi} 2^l l!
 *   \delta_{lk} = N_l \delta_{lk} \; .
 * \f]
 *
 * The recurrence relations is given by
 * \f[
 *   H_{l+1}(x) = 2 x H_l(x) - 2 l H_{l-1}(x) \; ,
 * \f]
 * with \f$ H_0(x) = 1 \f$ and \f$ H_1(x) = 2x \f$.
 *
 * For the derivative we have
 * \f[
 *   H'_{l+1}(x) = 2 (l + 1) H_l(x) \; ,
 * \f]
 * with \f$ H'_{0}(x) = 0 \f$ and \f$ H'_{1}(x) = 2 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Hermite_polynomials) for more information.
 */
class hermite_polynomial {
public:
    /**
     * @brief Get the domain \f$ \mathrm{D} \f$ on which the polynomials are defined.
     *
     * @return `std::array<double, 2>` representing the interval \f$ \mathrm{D} = (-\infty, \infty)
     * \f$.
     */
    [[nodiscard]] static constexpr auto domain() noexcept {
        return std::array<double, 2> { -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity() };
    }

    /**
     * @brief Get the normalization factor \f$ N_l = \langle H_l, H_l \rangle \f$ of the
     * l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor \f$ N_l = \sqrt{\pi} 2^l l! \f$.
     */
    [[nodiscard]] static constexpr double norm(int l) noexcept {
        return std::pow(2, l) * std::tgamma(l + 1) / std::numbers::inv_sqrtpi;
    }

    /**
     * @brief Get the value of the weight function \f$ W(x) \f$ evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Weight function \f$ W(x) = e^{-x^2} \f$.
     */
    [[nodiscard]] static constexpr double weight(double x) noexcept { return std::exp(-x * x); }

    /**
     * @brief Construct a sequence of Hermite polynomials for a specific \f$ x \f$ value.
     *
     * @details It sets \f$ x \f$ and initializes \f$ l = 0 \f$, \f$ H_0(x) = 1 \f$, \f$ H_1(x) = 2x
     * \f$, \f$ H'_0(x) = 0 \f$ and \f$ H'_1(x) = 2 \f$.
     *
     * @param x Value at which the polynomials and its derivatives are evaluated.
     */
    constexpr hermite_polynomial(double x = 0.0) noexcept : x_(x) {}

    /**
     * @brief Get the order \f$ l \f$ of the currently processed polynomial \f$ H_l(x) \f$.
     *
     * @return Order \f$ l \f$.
     */
    [[nodiscard]] constexpr int order() const noexcept { return l_; }

    /**
     * @brief Get the \f$ x \f$ value at which the polynomials are evaluated.
     *
     * @return \f$ x \f$ value given at construction.
     */
    [[nodiscard]] constexpr double x() const noexcept { return x_; }

    /**
     * @brief Get the function value of the currently processed polynomial.
     *
     * @return \f$ H_l(x) \f$.
     */
    [[nodiscard]] constexpr double current() const noexcept { return hl_; }

    /**
     * @brief Get the function value of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ H_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev() const noexcept { return hlm1_; }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return \f$ H'_l(x) \f$.
     */
    [[nodiscard]] constexpr double current_derivative() const noexcept { return dhl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ H'_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev_derivative() const noexcept { return dhlm1_; }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @details It applies the recurrence relations
     * \f[
     *   H_{l+1}(x) = 2 x H_l(x) - 2 l H_{l-1}(x)
     * \f]
     * and
     * \f[
     *   H'_{l+1}(x) = 2 (l + 1) H_l(x) \; .
     * \f]
     *
     * @return Function value of the polynomial \f$ H_l(x) \f$.
     */
    constexpr double next() noexcept {
        double hlp1 = 2.0 * x_;
        double dhlp1 = 2.0;
        if (l_ > 0) {
            hlp1 = 2.0 * x_ * hl_ - 2.0 * l_ * hlm1_;
            dhlp1 = 2.0 * (l_ + 1) * hl_;
        }
        hlm1_ = hl_;
        hl_ = hlp1;
        dhlm1_ = dhl_;
        dhl_ = dhlp1;
        ++l_;
        return hlm1_;
    }

private:
    int l_ { 0 };
    double x_;
    double hlm1_ { 0.0 };
    double hl_ { 1.0 };
    double dhlm1_ { 0.0 };
    double dhl_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_HERMITE_POLYNOMIAL_HPP
