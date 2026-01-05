/**
 * @file
 * @brief Implementation of Legendre polynomials using recurrence relations.
 */

#ifndef SIMPLEMC_NUMERIC_LEGENDRE_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_LEGENDRE_POLYNOMIAL_HPP

#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cassert>
#include <cmath>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-poly
 * @{
 */

/**
 * @brief Implementation of Legendre polynomials using recurrence relations.
 *
 * @details In the following, we use the notation from @ref simplemc-numeric-poly.
 *
 * Legendre polynomials, \f$ P_l(x) \f$, are defined on the interval \f$ \mathrm{D} = [-1, 1] \f$.
 * Their weight function is \f$ W(x) = 1 \f$ and their order dependent normalization factor is \f$ N_l
 * = 2 / (2l + 1) \f$:
 * \f[
 *   \langle P_l, P_k \rangle = \int_{-1}^1 P_l(x) P_k(x) dx = \frac{2}{2l + 1} \delta_{lk} = N_l
 *   \delta_{lk} \; .
 * \f]
 *
 * The recurrence relation is given by
 * \f[
 *   P_{l+1}(x) = \frac{(2l + 1) x P_l(x) - l P_{l-1}(x)}{l + 1} \; ,
 * \f]
 * with \f$ P_0(x) = 1 \f$ and \f$ P_1(x) = x \f$.
 *
 * For the derivative we have
 * \f[
 *   P'_{l+1}(x) = (l + 1) P_l(x) + x P'_{l}(x) \; ,
 * \f]
 * with \f$ P'_{0}(x) = 0 \f$ and \f$ P'_{1}(x) = 1 \f$.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Legendre_polynomials) for more information.
 */
class legendre_polynomial {
public:
    /**
     * @brief Get the domain \f$ \mathrm{D} \f$ on which the polynomials are defined.
     *
     * @return `std::array<double, 2>` representing the interval \f$ \mathrm{D} = [-1, 1] \f$.
     */
    [[nodiscard]] static constexpr auto domain() noexcept { return std::array<double, 2> { -1.0, 1.0 }; }

    /**
     * @brief Get the normalization factor \f$ N_l = \langle P_l, P_l \rangle \f$ of the
     * l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor \f$ N_l = 2 / (2 l + 1) \f$.
     */
    [[nodiscard]] static constexpr double norm(int l) noexcept { return 2.0 / (2.0 * l + 1); }

    /**
     * @brief Get the value of the weight function \f$ W(x) \f$ evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Weight function \f$ W(x) = 1 \f$.
     */
    [[nodiscard]] static constexpr double weight([[maybe_unused]] double x) noexcept { return 1.0; }

    /**
     * @brief Construct a sequence of Legendre polynomials for a specific \f$ x \f$ value.
     *
     * @details It sets \f$ x \f$ and initializes \f$ l = 0 \f$, \f$ P_0(x) = 1 \f$, \f$ P_1(x) = x
     * \f$, \f$ P'_0(x) = 0 \f$ and \f$ P'_1(x) = 1 \f$.
     *
     * It throws a simplemc::simplemc_exception if \f$ |x| > 1 \f$.
     *
     * @param x Value at which the polynomials and its derivatives are evaluated.
     */
    constexpr legendre_polynomial(double x = 0.0) : x_(x) {
        if (std::abs(x_) > 1) {
            throw simplemc_exception("Legendre polynomials are only defined on the interval [-1, 1]");
        }
    }

    /**
     * @brief Get the order \f$ l \f$ of the currently processed polynomial \f$ P_l(x) \f$.
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
     * @return \f$ P_l(x) \f$.
     */
    [[nodiscard]] constexpr double current() const noexcept { return pl_; }

    /**
     * @brief Get the function value of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ P_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev() const noexcept { return plm1_; }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return \f$ P'_l(x) \f$.
     */
    [[nodiscard]] constexpr double current_derivative() const noexcept { return dpl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @note The behaviour is undefined for \f$ l = 0 \f$.
     *
     * @return \f$ P'_{l-1}(x) \f$.
     */
    [[nodiscard]] constexpr double prev_derivative() const noexcept { return dplm1_; }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @details It applies the recurrence relations
     * \f[
     *   P_{l+1}(x) = \frac{(2l + 1) x P_l(x) - l P_{l-1}(x)}{l + 1}
     * \f]
     * and
     * \f[
     *   P'_{l+1}(x) = (l + 1) P_l(x) + x P'_{l}(x) \; .
     * \f]
     *
     * @return Function value of the polynomial \f$ P_l(x) \f$.
     */
    constexpr double next() noexcept {
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

private:
    int l_ { 0 };
    double x_;
    double plm1_ { 0.0 };
    double pl_ { 1.0 };
    double dplm1_ { 0.0 };
    double dpl_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LEGENDRE_POLYNOMIAL_HPP
