// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation of Chebyshev polynomials of the first and second kind using recurrence
 * relations.
 */

#ifndef SIMPLEMC_NUMERIC_CHEBYSHEV_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_CHEBYSHEV_POLYNOMIAL_HPP

#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <numbers>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-poly
 * @{
 */

/**
 * @brief Implementation of Chebyshev polynomials of the first kind using recurrence relations.
 *
 * @details In the following, we use the notation from @ref simplemc-numeric-poly.
 *
 * Chebyshev polynomials of the first kind, \f$ T_l(x) \f$, are defined on the interval \f$ \mathrm{D}
 * = [-1, 1] \f$. Their weight function is \f$ W(x) = 1 / \sqrt{1 - x^2} \f$ and their order dependent
 * normalization factor is \f$ N_l \f$:
 * \f[
 *   \langle T_l, T_k \rangle = \int_{-1}^1 T_l(x) T_k(x) \frac{1}{\sqrt{1 - x^2}} dx =
 *   \begin{cases}
 *   0 & \text{if } l \neq k \; , \\
 *   \pi & \text{if } l = k = 0 \; , \\
 *   \frac{\pi}{2} & \text{if } l = k \neq 0
 *   \end{cases}
 *   = N_l \delta_{lk} \; .
 * \f]
 *
 * The recurrence relation is given by
 * \f[
 *   T_{l+1}(x) = 2 x T_l(x) - T_{l-1}(x) \; ,
 * \f]
 * with \f$ T_0(x) = 1 \f$ and \f$ T_1(x) = x \f$.
 *
 * For the derivative we have
 * \f[
 *   T'_{l+1}(x) = 2 \left(T_l(x) + x T'_{l}(x) \right) - T'_{l-1}(x) \; ,
 * \f]
 * with \f$ T'_{0}(x) = 0 \f$ and \f$ T'_{1}(x) = 1 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials) for more information.
 */
class chebyshev_polynomial_first {
public:
    /**
     * @brief Get the domain \f$ \mathrm{D} \f$ on which the polynomials are defined.
     *
     * @return `std::array<double, 2>` representing the interval \f$ \mathrm{D} = [-1, 1] \f$.
     */
    [[nodiscard]] static constexpr auto domain() noexcept { return std::array<double, 2> { -1.0, 1.0 }; }

    /**
     * @brief Get the normalization factor \f$ N_l = \langle T_l, T_l \rangle \f$ of the l<sup>th
     * </sup> order polynomial.
     *
     * @details The normalization factor is given by
     * \f[
     *   N_l =
     *   \begin{cases}
     *   \pi & \text{if } l = 0 \; , \\
     *   \frac{\pi}{2} & \text{if } l \neq 0
     *   \end{cases}
     *   \; .
     * \f]
     *
     * @param l Order of the polynomial.
     * @return Normalization factor \f$ N_l \f$.
     */
    [[nodiscard]] static constexpr double norm(int l) noexcept {
        if (l == 0) {
            return std::numbers::pi;
        }
        return 0.5 * std::numbers::pi;
    }

    /**
     * @brief Get the value of the weight function \f$ W(x) \f$ evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Weight function \f$ W(x) = 1 / \sqrt{1 - x^2} \f$.
     */
    [[nodiscard]] static constexpr double weight(double x) noexcept { return 1.0 / std::sqrt(1 - x * x); }

    /**
     * @brief Construct a sequence of Chebyshev polynomials of the first kind for a specific \f$ x \f$
     * value.
     *
     * @details It sets \f$ x \f$ and initializes \f$ l = 0 \f$, \f$ T_0(x) = 1 \f$, \f$ T_1(x) = x
     * \f$, \f$ T'_0(x) = 0 \f$ and \f$ T'_1(x) = 1 \f$.
     *
     * It throws a simplemc::simplemc_exception if \f$ |x| > 1 \f$.
     *
     * @param x Value at which the polynomials and its derivatives are evaluated.
     */
    constexpr chebyshev_polynomial_first(double x = 0.0) : x_(x) {
        if (std::abs(x_) > 1) {
            throw simplemc_exception("Chebyshev polynomials are only defined on the interval [-1, 1]");
        }
    }

    /**
     * @brief Get the order \f$ l \f$ of the currently processed polynomial \f$ T_l(x) \f$.
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
     * @return \f$ T_l(x) \f$.
     */
    [[nodiscard]] constexpr double current() const noexcept { return tl_; }

    /**
     * @brief Get the function value of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ T_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev() const noexcept { return tlm1_; }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return \f$ T'_l(x) \f$.
     */
    [[nodiscard]] constexpr double current_derivative() const noexcept { return dtl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ T'_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev_derivative() const noexcept { return dtlm1_; }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @details It applies the recurrence relations
     * \f[
     *   T_{l+1}(x) = 2 x T_l(x) - T_{l-1}(x) \; ,
     * \f]
     * and
     * \f[
     *   T'_{l+1}(x) = 2 \left(T_l(x) + x T'_{l}(x) \right) - T'_{l-1}(x) \; .
     * \f]
     *
     * @return Function value of the polynomial \f$ T_l(x) \f$.
     */
    constexpr double next() noexcept {
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

private:
    int l_ { 0 };
    double x_;
    double tlm1_ { 0.0 };
    double tl_ { 1.0 };
    double dtlm1_ { 0.0 };
    double dtl_ { 0.0 };
};

/**
 * @brief Implementation of Chebyshev polynomials of the second kind using recurrence relations.
 *
 * @details In the following, we use the notation from @ref simplemc-numeric-poly.
 *
 * Chebyshev polynomials of the second kind, \f$ U_l(x) \f$, are defined on the interval \f$
 * \mathrm{D} = [-1, 1] \f$. Their weight function is \f$ W(x) = \sqrt{1 - x^2} \f$ and their order
 * dependent normalization factor is \f$ N_l = \pi / 2 \f$:
 * \f[
 *   \langle U_l, U_k \rangle = \int_{-1}^1 U_l(x) U_k(x) \sqrt{1 - x^2} dx = \frac{\pi}{2}
 *   \delta_{lk} = N_l \delta_{lk} \; .
 * \f]
 *
 * The recurrence relation is given by
 * \f[
 *   U_{l+1}(x) = 2x U_l(x) - U_{l-1}(x) \; ,
 * \f]
 * with \f$ U_0(x) = 1 \f$ and \f$ U_1(x) = 2 x \f$.
 *
 * For the derivative we have
 * \f[
 *   U'_{l+1}(x) = 2 \left(U_l(x) + x U'_{l}(x) \right) - U'_{l-1}(x) \; ,
 * \f]
 * with \f$ U'_{0}(x) = 0 \f$ and \f$ U'_{1}(x) = 2 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials) for more information.
 */
class chebyshev_polynomial_second {
public:
    /**
     * @brief Get the domain \f$ \mathrm{D} \f$ on which the polynomials are defined.
     *
     * @return `std::array<double, 2>` representing the interval \f$ \mathrm{D} = [-1, 1] \f$.
     */
    [[nodiscard]] static constexpr auto domain() noexcept { return std::array<double, 2> { -1.0, 1.0 }; }

    /**
     * @brief Get the normalization factor \f$ N_l = \langle U_l, U_l \rangle \f$ of the l<sup>th
     * </sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor \f$ N_l = \pi / 2 \f$.
     */
    [[nodiscard]] static constexpr double norm([[maybe_unused]] int l) noexcept { return 0.5 * std::numbers::pi; }

    /**
     * @brief Get the value of the weight function \f$ W(x) \f$ evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Weight function \f$ W(x) = \sqrt{1 - x^2} \f$.
     */
    [[nodiscard]] static constexpr double weight(double x) noexcept { return std::sqrt(1 - x * x); }

    /**
     * @brief Construct a sequence of Chebyshev polynomials of the second kind for a specific \f$ x
     * \f$ value.
     *
     * @details It sets \f$ x \f$ and initializes \f$ l = 0 \f$, \f$ U_0(x) = 1 \f$, \f$ U_1(x) = 2 x
     * \f$, \f$ U'_0(x) = 0 \f$ and \f$ U'_1(x) = 2 \f$.
     *
     * It throws a simplemc::simplemc_exception if \f$ |x| > 1 \f$.
     *
     * @param x Value at which the polynomials and its derivatives are evaluated.
     */
    constexpr chebyshev_polynomial_second(double x = 0.0) : x_(x) {
        if (std::abs(x_) > 1) {
            throw simplemc_exception("Chebyshev polynomials are only defined on the interval [-1, 1]");
        }
    }

    /**
     * @brief Get the order \f$ l \f$ of the currently processed polynomial \f$ U_l(x) \f$.
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
     * @return \f$ U_l(x) \f$.
     */
    [[nodiscard]] constexpr double current() const noexcept { return ul_; }

    /**
     * @brief Get the function value of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ U_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev() const noexcept { return ulm1_; }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return \f$ U'_l(x) \f$.
     */
    [[nodiscard]] constexpr double current_derivative() const noexcept { return dul_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ U'_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev_derivative() const noexcept { return dulm1_; }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @details It applies the recurrence relations
     * \f[
     *   U_{l+1}(x) = 2x U_l(x) - U_{l-1}(x) \; ,
     * \f]
     * and
     * \f[
     *   U'_{l+1}(x) = 2 \left(U_l(x) + x U'_{l}(x) \right) - U'_{l-1}(x) \; .
     * \f]
     *
     * @return Function value of the polynomial \f$ U_l(x) \f$.
     */
    constexpr double next() noexcept {
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

private:
    int l_ { 0 };
    double x_;
    double ulm1_ { 0.0 };
    double ul_ { 1.0 };
    double dulm1_ { 0.0 };
    double dul_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_CHEBYSHEV_POLYNOMIAL_HPP
