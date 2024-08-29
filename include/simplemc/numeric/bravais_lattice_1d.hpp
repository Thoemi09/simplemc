/**
 * @file
 * @brief Bravais lattices in 1D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_1D_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_1D_HPP

#include <simplemc/numeric/bravais_lattice.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices-1d
 * @{
 */

/**
 * @brief Check 1D lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_1d_lattice_parameters(const lattice_parameters& p);

/**
 * @brief Make a linear Bravais lattice in 1D.
 *
 * @details A linear Bravais lattice is defined by
 * - the lattice constant \f$ a \f$.
 *
 * @param a Lattice constant.
 * @return Linear lattice.
 */
[[nodiscard]] bravais_lattice<1> make_linear_lattice(double a);

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_1D_HPP
