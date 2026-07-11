// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Bravais lattices in 2D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_2D_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_2D_HPP

#include <simplemc/numeric/bravais_lattice.hpp>

#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices-2d
 * @{
 */

namespace detail {

// Check the lattice parameters for 2D lattices.
void check_2d_lattice_parameters(const lattice_parameters& p);

} // namespace detail

/**
 * @brief Make a square Bravais lattice in 2D.
 *
 * @details A square Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<2>, lattice_parameters> make_square_lattice(double a);

/**
 * @brief Make a hexagonal Bravais lattice in 2D.
 *
 * @details A hexagonal Bravais lattice in 2D is defined by
 * - the lattice constant \f$ a > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$ and
 * - the angle \f$ \gamma = 2 \pi / 3 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$.
 *
 * @param a Lattice constant.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<2>, lattice_parameters> make_hexagonal_lattice(double a);

/**
 * @brief Make a rectangular Bravais lattice in 2D.
 *
 * @details A rectangular Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$ and
 * - the lattice constant \f$ b > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$ or \f$ b \leq 0 \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<2>, lattice_parameters> make_rectangular_lattice(double a, double b);

/**
 * @brief Make a rectangular centered Bravais lattice in 2D.
 *
 * @details A rectangular centered Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$ and
 * - the lattice constant \f$ b > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$ or \f$ b \leq 0 \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<2>, lattice_parameters> make_rectangular_centered_lattice(double a, double b);

/**
 * @brief Make an oblique Bravais lattice in 2D.
 *
 * @details An oblique Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$ and
 * - the angle \f$ \gamma \in (0, \pi] \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$ or \f$ \gamma
 * \notin (0, \pi] \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param gamma Angle \f$ \gamma \in (0, \pi] \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<2>, lattice_parameters> make_oblique_lattice(double a, double b, double gamma);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_2D_HPP
