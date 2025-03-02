/**
 * @file
 * @brief Bravais lattices in 1D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_1D_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_1D_HPP

#include <simplemc/numeric/bravais_lattice.hpp>

#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices-1d
 * @{
 */

namespace detail {

// Check the lattice parameters for 1D lattices.
void check_1d_lattice_parameters(const lattice_parameters& p);

} // namespace detail

/**
 * @brief Make a linear Bravais lattice in 1D.
 *
 * @details A linear Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<1>, lattice_parameters> make_linear_lattice(double a);

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_1D_HPP
