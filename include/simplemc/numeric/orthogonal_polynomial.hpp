// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Concept and utility functions for orthogonal polynomials.
 */

#ifndef SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIAL_HPP

#include <array>
#include <concepts>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-poly
 * @{
 */

/**
 * @brief Concept that specifies the requirements for a sequence of orthogonal polynomials.
 *
 * @details We can directly map the requirements to the notation used in @ref simplemc-numeric-poly.
 * Let `P` be the type of the orthogonal polynomial and let `p` be an instance of `P`.
 *
 * - `P(double)` constructs a new instance `p` for a specific \f$ x \f$ and initializes \f$ p_l(x) =
 * p_0(x) \f$.
 * - `P::domain()` returns the domain \f$ \mathrm{D} = [a, b] \f$ on which the polynomials \f$ p_l(x)
 * \f$ are defined.
 * - `P::norm(int)` returns the normalization \f$ N_l \f$ for a given order \f$ l \f$.
 * - `P::weight(double)` returns the value of the weight function \f$ W(x) \f$ at a given \f$ x \f$.
 * - `p.next()` increases the order from \f$ l \f$ to \f$ l + 1 \f$ and returns \f$ p_{l}(x) \f$.
 */
template <typename P>
concept orthogonal_polynomial = requires(P& p, const P& cp, double x, int l) {
    P(x);
    { P::domain() } -> std::same_as<std::array<double, 2>>;
    { P::norm(l) } -> std::same_as<double>;
    { P::weight(x) } -> std::same_as<double>;
    { p.next() } -> std::same_as<double>;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIAL_HPP
