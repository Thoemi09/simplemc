/**
 * @file
 * @brief Bravais lattices in 2D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_2D_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_2D_HPP

#include <simplemc/numeric/bravais_lattice.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices-2d
 * @{
 */

/**
 * @brief Check 2D lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_2d_lattice_parameters(const lattice_parameters& p);

/**
 * @brief Make a square Bravais lattice in 2D.
 *
 * @details A square Bravais lattice is defined by
 * - the lattice constant \f$ a \f$.
 *
 * @param a Lattice constant.
 * @return Square lattice.
 */
[[nodiscard]] bravais_lattice<2> make_square_lattice(double a);

/**
 * @brief Make a hexagonal Bravais lattice in 2D.
 *
 * @details A hexagonal Bravais lattice in 2D is defined by
 * - the lattice constant \f$ a \f$.
 *
 * @param a Lattice constant.
 * @return 2D hexagonal lattice.
 */
[[nodiscard]] bravais_lattice<2> make_hexagonal_lattice(double a);

/**
 * @brief Make a rectangular Bravais lattice in 2D.
 *
 * @details A rectangular Bravais lattice is defined by
 * - the lattice constant \f$ a \f$ and
 * - the lattice constant \f$ b \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @return Rectangular lattice.
 */
[[nodiscard]] bravais_lattice<2> make_rectangular_lattice(double a, double b);

/**
 * @brief Make a rectangular centered Bravais lattice in 2D.
 *
 * @details A rectangular centered Bravais lattice is defined by
 * - the lattice constant \f$ a \f$ and
 * - the lattice constant \f$ b \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @return Rectangular centered lattice.
 */
[[nodiscard]] bravais_lattice<2> make_rectangular_centered_lattice(double a, double b);

/**
 * @brief Make an oblique Bravais lattice in 2D.
 *
 * @details An oblique Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$ and
 * - the angle \f$ \gamma \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param gamma Angle between the two directions.
 * @return Oblique lattice.
 */
[[nodiscard]] bravais_lattice<2> make_oblique_lattice(double a, double b, double gamma);

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_2D_HPP
