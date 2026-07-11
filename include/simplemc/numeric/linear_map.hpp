// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Linear map from the domain \f$ [a, b] \f$ to the range \f$ [c, d] \f$ (and vice versa).
 */

#ifndef SIMPLEMC_NUMERIC_LINEAR_MAP_HPP
#define SIMPLEMC_NUMERIC_LINEAR_MAP_HPP

#include <simplemc/utils/simplemc_exception.hpp>

#include <cassert>
#include <array>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-utils
 * @{
 */

/**
 * @brief Map a domain \f$ [a, b] \f$ to the range \f$ [c, d] \f$ (and vice versa) via a linear
 * function \f$ y : [a, b] \in \mathbb{R} \to [c, d] \in \mathbb{R} \f$.
 *
 * @details The linear function is defined by
 * \f[
 *   y(x) = \alpha * x + \beta
 * \f]
 * such that \f$ y(a) = c \f$ and \f$ y(b) = d \f$.
 *
 * Its inverse is then given by
 * \f[
 *   y^{-1}(x) =  \frac{x - \beta}{\alpha}
 * \f]
 * such that \f$ y^{-1}(c) = a \f$ and \f$ y^{-1}(d) = b \f$.
 *
 * It is assumed that \f$ a < b \f$ and \f$ c < d \f$.
 */
class linear_map {
public:
    /**
     * @brief Type specifying an interval.
     */
    using interval_type = std::array<double, 2>;

    /**
     * @brief Default constructor leaves the map uninitialized.
     */
    constexpr linear_map() noexcept = default;

    /**
     * @brief Construct the linear map \f$ y : [a, b] \to [c, d] \f$.
     *
     * @param dom Domain \f$ [a, b] \f$ as an interval_type.
     * @param rg Range \f$ [c, d] \f$ as an interval_type.
     */
    linear_map(const interval_type& dom, const interval_type& rg) { set(dom, rg); }

    /**
     * @brief Set the domain and the range of the linear map.
     *
     * @details It throws a simplemc::simplemc_exception if the domain, range or the resulting map is
     * illdefined.
     *
     * @param dom Domain \f$ [a, b] \f$ as an interval_type.
     * @param rg Range \f$ [c, d] \f$ as an interval_type.
     */
    constexpr void set(const interval_type& dom, const interval_type& rg) {
        constexpr auto inf = std::numeric_limits<double>::infinity();
        constexpr auto minus_inf = -inf;
        double a = dom[0];
        double b = dom[1];
        double c = rg[0];
        double d = rg[1];
        if (a >= b || c >= d) {
            throw simplemc_exception("Wrong intervals in linear map", "linear_map::set");
        }
        if (a > minus_inf && b < inf && c > minus_inf && d < inf) {
            alpha_ = (c - d) / (a - b);
            beta_ = c - a * alpha_;
        } else if (a == minus_inf && b != inf && c == minus_inf && d != inf) {
            alpha_ = 1.0;
            beta_ = d - b;
        } else if (a != minus_inf && b == inf && c != minus_inf && d == inf) {
            alpha_ = 1.0;
            beta_ = c - a;
        } else if (a == minus_inf && b == inf && c == minus_inf && d == inf) {
            alpha_ = 1.0;
            beta_ = 0.0;
        } else {
            throw simplemc_exception("Wrong intervals in linear map", "linear_map::set");
        }
        domain_ = dom;
        range_ = rg;
    }

    /**
     * @brief Map a value \f$ x \f$ from the domain \f$ [a, b] \f$ to the range \f$ [c, d] \f$.
     *
     * @param x Input value \f$ x \in [a, b] \f$.
     * @return Mapped value \f$ y(x) \in [c, d] \f$.
     */
    [[nodiscard]] constexpr double map(double x) const {
        assert(x >= domain_[0] && x <= domain_[1]);
        return alpha_ * x + beta_;
    }

    /**
     * @brief Use the inverse function to map a value \f$ x \f$ from the original range \f$ [c, d] \f$
     * to the original domain \f$ [a, b] \f$.
     *
     * @param x Input value \f$ x \in [c, d] \f$.
     * @return Mapped value \f$ y^{-1}(x) \in [a, b] \f$.
     */
    [[nodiscard]] constexpr double inverse_map(double x) const {
        assert(x >= range_[0] && x <= range_[1]);
        return (x - beta_) / alpha_;
    }

    /**
     * @brief Get the domain of the linear map.
     *
     * @return Domain \f$ [a, b] \f$ as an interval_type.
     */
    [[nodiscard]] constexpr auto domain() const noexcept { return domain_; }

    /**
     * @brief Get the range of the linear map.
     *
     * @return Range \f$ [c, d] \f$ as an interval_type.
     */
    [[nodiscard]] constexpr auto range() const noexcept { return range_; }

    /**
     * @brief Get the linear function parameters.
     *
     * @return `std::pair` containing the parameters \f$ \alpha \f$ and \f$ \beta \f$.
     */
    [[nodiscard]] constexpr auto params() const noexcept { return std::make_pair(alpha_, beta_); }

private:
    interval_type domain_ { 0.0, 1.0 };
    interval_type range_ { 0.0, 1.0 };
    double alpha_ { 1.0 };
    double beta_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LINEAR_MAP_HPP
