// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation of Laguerre polynomials using recurrence relations.
 */

#ifndef SIMPLEMC_NUMERIC_LAGUERRE_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_LAGUERRE_POLYNOMIAL_HPP

#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-poly
 * @{
 */

/**
 * @brief Implementation of Laguerre polynomials using recurrence relations.
 *
 * @details In the following, we use the notation from @ref simplemc-numeric-poly.
 *
 * Laguerre polynomials, \f$ L_l(x) \f$, are defined on the interval \f$ \mathrm{D} = [0, \infty) \f$.
 * Their weight function is \f$ W(x) = e^{-x} \f$ and the constant normalization factor is \f$ N_l = 1
 * \f$:
 * \f[
 *   \langle L_l, L_k \rangle = \int_0^\infty L_l(x) L_k(x) e^{-x} dx = \delta_{lk} = N_l \delta_{lk}
 *   \; .
 * \f]
 *
 * The recurrence relations is given by
 * \f[
 *   L_{l+1}(x) = \frac{(2l + 1 - x) L_l(x) - l L_{l-1}(x)}{l + 1} \; ,
 * \f]
 * with \f$ L_0(x) = 1 \f$ and \f$ L_1(x) = 1 - x \f$.
 *
 * For the derivative we have
 * \f[
 *   L'_{l+1}(x) = \frac{(2l + 1 - x) L'_l(x) - L_l(x) - l L'_{l-1}(x)}{l + 1} \; ,
 * \f]
 * with \f$ L'_{0}(x) = 0 \f$ and \f$ L'_{1}(x) = -1 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Laguerre_polynomials) for more information.
 */
class laguerre_polynomial {
public:
    /**
     * @brief Get the domain \f$ \mathrm{D} \f$ on which the polynomials are defined.
     *
     * @return `std::array<double, 2>` representing the interval \f$ \mathrm{D} = [0, \infty)
     * \f$.
     */
    [[nodiscard]] static constexpr auto domain() noexcept {
        return std::array<double, 2> { 0, std::numeric_limits<double>::infinity() };
    }

    /**
     * @brief Get the normalization factor \f$ N_l = \langle L_l, L_l \rangle \f$ of the
     * l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor \f$ N_l = 1 \f$.
     */
    [[nodiscard]] static constexpr double norm([[maybe_unused]] int l) noexcept { return 1.0; }

    /**
     * @brief Get the value of the weight function \f$ W(x) \f$ evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Weight function \f$ W(x) = e^{-x} \f$.
     */
    [[nodiscard]] static constexpr double weight(double x) noexcept { return std::exp(-x); }

    /**
     * @brief Construct a sequence of Laguerre polynomials for a specific \f$ x \f$ value.
     *
     * @details It sets \f$ x \f$ and initializes \f$ l = 0 \f$, \f$ L_0(x) = 1 \f$, \f$ L_1(x) = 1 -
     * x \f$, \f$ L'_0(x) = 0 \f$ and \f$ L'_1(x) = -1 \f$.
     *
     * It throws a simplemc::simplemc_exception if \f$ x < 0 \f$.
     *
     * @param x Value at which the polynomials and its derivatives are evaluated.
     */
    constexpr laguerre_polynomial(double x = 0.0) : x_(x) {
        if (x_ < 0) {
            throw simplemc_exception("Laguerre polynomials are only defined on the interval [0, inf)");
        }
    }

    /**
     * @brief Get the order \f$ l \f$ of the currently processed polynomial \f$ L_l(x) \f$.
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
     * @return \f$ L_l(x) \f$.
     */
    [[nodiscard]] constexpr double current() const noexcept { return ll_; }

    /**
     * @brief Get the function value of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ L_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev() const noexcept { return llm1_; }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return \f$ L'_l(x) \f$.
     */
    [[nodiscard]] constexpr double current_derivative() const noexcept { return dll_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ L'_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev_derivative() const noexcept { return dllm1_; }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @details It applies the recurrence relations
     * \f[
     *   L_{l+1}(x) = \frac{(2l + 1 - x) L_l(x) - l L_{l-1}(x)}{l + 1}
     * \f]
     * and
     * \f[
     *   L'_{l+1}(x) = \frac{(2l + 1 - x) L'_l(x) - L_l(x) - l L'_{l-1}(x)}{l + 1} \; .
     * \f]
     *
     * @return Function value of the polynomial \f$ L_l(x) \f$.
     */
    constexpr double next() noexcept {
        double llp1 = 1.0 - x_;
        double dllp1 = -1.0;
        if (l_ > 0) {
            const double a = -1.0 / (l_ + 1);
            const double b = 2.0 + a;
            const double c = 1.0 + a;
            llp1 = (a * x_ + b) * ll_ - c * llm1_;
            dllp1 = a * ll_ + (a * x_ + b) * dll_ - c * dllm1_;
        }
        llm1_ = ll_;
        ll_ = llp1;
        dllm1_ = dll_;
        dll_ = dllp1;
        ++l_;
        return llm1_;
    }

private:
    int l_ { 0 };
    double x_;
    double llm1_ { 0.0 };
    double ll_ { 1.0 };
    double dllm1_ { 0.0 };
    double dll_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LAGUERRE_POLYNOMIAL_HPP
