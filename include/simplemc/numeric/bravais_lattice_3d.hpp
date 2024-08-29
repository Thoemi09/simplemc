/**
 * @file
 * @brief Bravais lattices in 3D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_3D_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_3D_HPP

#include <simplemc/numeric/bravais_lattice.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices-3d
 * @{
 */

/**
 * @brief Check 3D lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_3d_lattice_parameters(const lattice_parameters& p);

/**
 * @brief Make a cubic Bravais lattice in 3D.
 *
 * @details A cubic Bravais lattice is defined by
 * - the lattice constant \f$ a \f$.
 *
 * @param a Lattice constant.
 * @return Cubic lattice.
 */
[[nodiscard]] bravais_lattice<3> make_cubic_lattice(double a);

/**
 * @brief Make an FCC Bravais lattice in 3D.
 *
 * @details An FCC Bravais lattice is defined by
 * - the lattice constant \f$ a \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant.
 * @return FCC lattice.
 */
[[nodiscard]] bravais_lattice<3> make_fcc_lattice(double a);

/**
 * @brief Make a BCC Bravais lattice in 3D.
 *
 * @details A BCC Bravais lattice is defined by
 * - the lattice constant \f$ a \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant.
 * @return BCC lattice.
 */
[[nodiscard]] bravais_lattice<3> make_bcc_lattice(double a);

/**
 * @brief Make a hexagonal Bravais lattice in 3D.
 *
 * @details A hexagonal Bravais lattice in 3D is defined by
 * - the lattice constant \f$ a \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param c Lattice constant in direction #3.
 * @return Hexagonal lattice.
 */
[[nodiscard]] bravais_lattice<3> make_hexagonal_lattice(double a, double c);

/**
 * @brief Make a rhombohedral Bravais lattice in 3D.
 *
 * @details A rhombohedral Bravais lattice is defined by
 * - the lattice constant \f$ a \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param c Lattice constant in direction #3.
 * @return Rhombohedral lattice.
 */
[[nodiscard]] bravais_lattice<3> make_rhombohedral_lattice(double a, double c);

/**
 * @brief Make a tetragonal Bravais lattice in 3D.
 *
 * @details A tetragonal Bravais lattice is defined by
 * - the lattice constant \f$ a \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param c Lattice constant in direction #3.
 * @return Tetragonal lattice.
 */
[[nodiscard]] bravais_lattice<3> make_tetragonal_lattice(double a, double c);

/**
 * @brief Make a tetragonal body-centered Bravais lattice in 3D.
 *
 * @details A tetragonal body-centered Bravais lattice is defined by
 * - the lattice constant \f$ a \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param c Lattice constant in direction #3.
 * @return Tetragonal body-centered lattice.
 */
[[nodiscard]] bravais_lattice<3> make_tetragonal_bc_lattice(double a, double c);

/**
 * @brief Make an orthorhombic Bravais lattice in 3D.
 *
 * @details An orthorhombic Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @return Orthorhombic lattice.
 */
[[nodiscard]] bravais_lattice<3> make_orthorhombic_lattice(double a, double b, double c);

/**
 * @brief Make an orthorombic body-centered Bravais lattice in 3D.
 *
 * @details An orthorhombic body-centered Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @return Orthorhombic body-centered lattice.
 */
[[nodiscard]] bravais_lattice<3> make_orthorhombic_bc_lattice(double a, double b, double c);

/**
 * @brief Make an orthorombic face-centered Bravais lattice in 3D.
 *
 * @details An orthorhombic face-centered Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @return Orthorhombic face-centered lattice.
 */
[[nodiscard]] bravais_lattice<3> make_orthorhombic_fc_lattice(double a, double b, double c);

/**
 * @brief Make an orthorombic base-centered Bravais lattice in 3D.
 *
 * @details An orthorhombic base-centered Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$ and
 * - the lattice constant \f$ c \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @return Orthorhombic base-centered lattice.
 */
[[nodiscard]] bravais_lattice<3> make_orthorhombic_base_centered_lattice(double a, double b, double c);

/**
 * @brief Make a monoclinic Bravais lattice in 3D.
 *
 * @details A monoclinic Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$,
 * - the lattice constant \f$ c \f$ and
 * - the angle \f$ \beta \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @param beta Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$.
 * @return Monoclinic lattice.
 */
[[nodiscard]] bravais_lattice<3> make_monoclinic_lattice(double a, double b, double c, double beta);

/**
 * @brief Make a monoclinic base-centered Bravais lattice in 3D.
 *
 * @details A monoclinic base-centered Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$,
 * - the lattice constant \f$ c \f$ and
 * - the angle \f$ \beta \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$.
 *
 * @note The parameters correspond to the conventional and not the primitive unit cell.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @param beta Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$.
 * @return Monoclinic base-centered lattice.
 */
[[nodiscard]] bravais_lattice<3> make_monoclinic_base_centered_lattice(double a, double b, double c, double beta);

/**
 * @brief Make a triclinic Bravais lattice in 3D.
 *
 * @details A triclinic Bravais lattice is defined by
 * - the lattice constant \f$ a \f$,
 * - the lattice constant \f$ b \f$,
 * - the lattice constant \f$ c \f$,
 * - the angle \f$ \alpha \f$ between \f$ \mathbf{b} \f$ and \f$ \mathbf{c} \f$,
 * - the angle \f$ \beta \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$ and
 * - the angle \f$ \gamma \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$.
 *
 * @param a Lattice constant in direction #1.
 * @param b Lattice constant in direction #2.
 * @param c Lattice constant in direction #3.
 * @param alpha Angle between \f$ \mathbf{b} \f$ and \f$ \mathbf{c} \f$.
 * @param beta Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$.
 * @param gamma Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$.
 * @return Triclinic lattice.
 */
[[nodiscard]] bravais_lattice<3> make_triclinic_lattice(
    double a, double b, double c, double alpha, double beta, double gamma);

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_3D_HPP
