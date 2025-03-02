/**
 * @file
 * @brief Bravais lattices in 3D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_3D_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_3D_HPP

#include <simplemc/numeric/bravais_lattice.hpp>

#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices-3d
 * @{
 */

namespace detail {

// Check the lattice parameters for 3D lattices.
void check_3d_lattice_parameters(const lattice_parameters& p);

} // namespace detail

/**
 * @brief Make a cubic Bravais lattice in 3D.
 *
 * @details A cubic Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the lattice constant \f$ c = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_cubic_lattice(double a);

/**
 * @brief Make an FCC Bravais lattice in 3D.
 *
 * @details An FCC Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the lattice constant \f$ c = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_fcc_lattice(double a);

/**
 * @brief Make a BCC Bravais lattice in 3D.
 *
 * @details A BCC Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the lattice constant \f$ c = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_bcc_lattice(double a);

/**
 * @brief Make a hexagonal Bravais lattice in 3D.
 *
 * @details A hexagonal Bravais lattice in 3D is defined by
 * - the lattice constant \f$ a > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = 2 \pi / 3 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$ or \f$ c \leq 0 \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_hexagonal_lattice(double a, double c);

/**
 * @brief Make a rhombohedral Bravais lattice in 3D.
 *
 * @details A rhombohedral Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = 2 \pi / 3 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$ or \f$ c \leq 0 \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_rhombohedral_lattice(double a, double c);

/**
 * @brief Make a tetragonal Bravais lattice in 3D.
 *
 * @details A tetragonal Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$ or \f$ c \leq 0 \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_tetragonal_lattice(double a, double c);

/**
 * @brief Make a tetragonal body-centered Bravais lattice in 3D.
 *
 * @details A tetragonal body-centered Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the lattice constant \f$ b = a \f$,
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$ or \f$ c \leq 0 \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_tetragonal_bc_lattice(double a, double c);

/**
 * @brief Make an orthorhombic Bravais lattice in 3D.
 *
 * @details An orthorhombic Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$ or \f$ c \leq 0
 * \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_lattice(double a, double b, double c);

/**
 * @brief Make an orthorombic body-centered Bravais lattice in 3D.
 *
 * @details An orthorhombic body-centered Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$ or \f$ c \leq 0
 * \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_bc_lattice(
    double a, double b, double c);

/**
 * @brief Make an orthorombic face-centered Bravais lattice in 3D.
 *
 * @details An orthorhombic face-centered Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$ or \f$ c \leq 0
 * \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_fc_lattice(
    double a, double b, double c);

/**
 * @brief Make an orthorombic base-centered Bravais lattice in 3D.
 *
 * @details An orthorhombic base-centered Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$ and
 * - the lattice constant \f$ c > 0 \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \alpha = \pi / 2 \f$,
 * - the angle \f$ \beta = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$ or \f$ c \leq 0
 * \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_base_centered_lattice(
    double a, double b, double c);

/**
 * @brief Make a monoclinic Bravais lattice in 3D.
 *
 * @details A monoclinic Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$,
 * - the lattice constant \f$ c > 0 \f$ and
 * - the angle \f$ \beta \in (0, \pi] \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \alpha = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$, \f$ c \leq 0 \f$
 * or \f$ \beta \notin (0, \pi] \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @param beta Angle \f$ \beta \in (0, \pi] \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_monoclinic_lattice(
    double a, double b, double c, double beta);

/**
 * @brief Make a monoclinic base-centered Bravais lattice in 3D.
 *
 * @details A monoclinic base-centered Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$,
 * - the lattice constant \f$ c > 0 \f$ and
 * - the angle \f$ \beta \in (0, \pi] \f$.
 *
 * The following lattice parameters are set as well:
 * - the angle \f$ \alpha = \pi / 2 \f$ and
 * - the angle \f$ \gamma = \pi / 2 \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$, \f$ c \leq 0 \f$
 * or \f$ \beta \notin (0, \pi] \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell. The primitive
 * unit cell vectors are stored in the returned simplemc::bravais_lattice object.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @param beta Angle \f$ \beta \in (0, \pi] \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_monoclinic_base_centered_lattice(
    double a, double b, double c, double beta);

/**
 * @brief Make a triclinic Bravais lattice in 3D.
 *
 * @details A triclinic Bravais lattice is defined by
 * - the lattice constant \f$ a > 0 \f$,
 * - the lattice constant \f$ b > 0 \f$,
 * - the lattice constant \f$ c > 0 \f$,
 * - the angle \f$ \alpha \in (0, \pi] \f$,
 * - the angle \f$ \beta \in (0, \pi] \f$ and
 * - the angle \f$ \gamma \in (0, \pi] \f$.
 *
 * It throws a simplemc::simplemc_exception, if \f$ a \leq 0 \f$, \f$ b \leq 0 \f$, \f$ c \leq 0 \f$,
 * \f$ \alpha \notin (0, \pi] \f$, \f$ \beta \notin (0, \pi] \f$ or \f$ \gamma \notin (0, \pi] \f$.
 *
 * @param a Lattice constant \f$ a > 0 \f$.
 * @param b Lattice constant \f$ b > 0 \f$.
 * @param c Lattice constant \f$ c > 0 \f$.
 * @param alpha Angle \f$ \alpha \in (0, \pi] \f$.
 * @param beta Angle \f$ \beta \in (0, \pi] \f$.
 * @param gamma Angle \f$ \gamma \in (0, \pi] \f$.
 * @return `std::pair` containing the created simplemc::bravais_lattice and the
 * simplemc::lattice_parameters used to construct the lattice.
 */
[[nodiscard]] std::pair<bravais_lattice<3>, lattice_parameters> make_triclinic_lattice(
    double a, double b, double c, double alpha, double beta, double gamma);

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_3D_HPP
