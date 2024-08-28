/**
 * @file
 * @brief Bravais lattices in 1D, 2D and 3D.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP

#include <simplemc/numeric/eigen.hpp>

#include <string>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices
 * @{
 */

/**
 * @brief Class representing a Bravais lattice in 1D, 2D and 3D.
 *
 * @details A lattice is defined by \f$ N \f$ N-dimensional lattice vectors. One lattice point is in
 * the origin and all others are obtained by translating the origin using linear combinations of the
 * lattice vectors.
 *
 * The lattice can be in 1, 2 or 3 dimensions. Other dimensions are not supported.
 *
 * It is assumed that the lattice vectors are linearly independent and that the lattice parameters are
 * correctly set. It is recommended to use the provided factory functions instead of manually creating
 * a lattice (see @ref simplemc-numeric-lattices-1d, @ref simplemc-numeric-lattices-2d and
 * @ref simplemc-numeric-lattices-3d).
 */
class bravais_lattice {
public:
    /**
     * @brief Type of vector used.
     */
    using vector_type = Eigen::VectorXd;

    /**
     * @brief Type of matrix used.
     */
    using matrix_type = Eigen::MatrixXd;

    /**
     * @brief Tags for the different Bravais lattices.
     */
    enum class lattice_tag {
        linear_1d,
        square_2d,
        hexagonal_2d,
        rectangular_2d,
        rectangular_centered_2d,
        oblique_2d,
        cubic_3d,
        fcc_3d,
        bcc_3d,
        hexagonal_3d,
        rhombohedral_3d,
        tetragonal_3d,
        tetragonal_bc_3d,
        orthorhombic_3d,
        orthorhombic_fc_3d,
        orthorhombic_bc_3d,
        orthorhombic_base_centered_3d,
        monoclinic_3d,
        monoclinic_base_centered_3d,
        triclinic_3d,
        unspecified
    };

    /**
     * @brief Parameters specifying a Bravais lattice.
     *
     * @details The following parameters are stored:
     *
     * - \f$ a \f$: Lattice constant in direction #1.
     * - \f$ b \f$: Lattice constant in direction #2 (only in 2D and 3D).
     * - \f$ c \f$: Lattice constant in direction #3 (only in 3D).
     * - \f$ \alpha \f$: Angle between \f$ \mathbf{b} \f$ and \f$ \mathbf{c} \f$ (only in 3D).
     * - \f$ \beta \f$: Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$ (only in 3D).
     * - \f$ \gamma \f$: Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$ (in 2D and 3D).
     * - `tag`: Tag specifying the type of Bravais lattice.
     */
    struct params {
        double a { 0.0 };
        double b { 0.0 };
        double c { 0.0 };
        double alpha { 0.0 };
        double beta { 0.0 };
        double gamma { 0.0 };
        lattice_tag tag { lattice_tag::unspecified };
    };

    /**
     * @brief Check dimensions of the lattice vector matrix.
     *
     * @details It throws an exception if the dimensions are not supported, i.e. if the matrix is
     * neither \f$ 1 \times 1 \f$, \f$ 2 \times 2 \f$ or \f$ 3 \times 3 \f$.
     *
     * @param mat Matrix with the lattice vectors in its columns.
     */
    static void check_lattice_vector_dims(const matrix_type& mat);

    /**
     * @brief Calculate the volume of a given cell.
     *
     * @param mat Matrix with the lattice vectors in its columns.
     * @return Volume spanned by the lattice vectors.
     */
    [[nodiscard]] static double calculate_cell_volume(const matrix_type& mat);

    /**
     * @brief Calculate the reciprocal lattice vectors w.r.t. to a given lattice.
     *
     * @param mat Matrix with the lattice vectors in its columns.
     * @return Matrix with the corresponding reciprocal lattice vectors in its columns.
     */
    [[nodiscard]] static matrix_type calculate_reciprocal_lattice(const matrix_type& mat);

    /**
     * @brief Calculate the vertices of the cell spanned by given lattice vectors.
     *
     * @details The following matrix sizes are returned:
     * - 1D: \f$ 1 \times 2 \f$
     * - 2D: \f$ 2 \times 4 \f$
     * - 3D: \f$ 3 \times 8 \f$
     *
     * @param mat Matrix with the lattice vectors in its columns.
     * @return Matrix with the vertices in its columns.
     */
    [[nodiscard]] static matrix_type calculate_cell_vertices(const matrix_type& mat);

    /**
     * @brief Default constructor leaves the Bravais lattice unspecified.
     */
    bravais_lattice() = default;

    /**
     * @brief Construct a Bravais lattice with lattice vectors and parameters.
     *
     * @warning No checks are performed that the lattice vectors and parameters are consistent.
     *
     * @param mat Matrix with the lattice vectors in its columns.
     * @param p Bravais lattice parameters.
     */
    bravais_lattice(matrix_type mat, const params& p);

    /**
     * @brief Get the number of dimensions of the Bravais lattice.
     */
    [[nodiscard]] auto dim() const { return real_lat_.rows(); };

    /**
     * @brief Get the Bravais lattice parameters.
     *
     * @return Bravais lattice parameters.
     */
    [[nodiscard]] const params& get_lattice_parameters() const { return params_; }

    /**
     * @brief Get the real space lattice vectors.
     *
     * @return Matrix with the real space lattice vectors in its columns.
     */
    [[nodiscard]] const matrix_type& real_lattice_vectors() const { return real_lat_; }

    /**
     * @brief Get the unit cell volume spanned by the real space lattice vectors.
     *
     * @return Unit cell volume in real space.
     */
    [[nodiscard]] double real_cell_volume() const { return real_vol_; }

    /**
     * @brief Get the reciprocal space lattice vectors.
     *
     * @return Matrix with the reciprocal space lattice vectors in its columns.
     */
    [[nodiscard]] const matrix_type& reciprocal_lattice_vectors() const { return rec_lat_; }

    /**
     * @brief Get the unit cell volume spanned by the reciprocal space lattice vectors.
     *
     * @return Unit cell volume in reciprocal space.
     */
    [[nodiscard]] double reciprocal_cell_volume() const { return rec_vol_; }

private:
    params params_;
    matrix_type real_lat_;
    matrix_type rec_lat_;
    double real_vol_ { 0.0 };
    double rec_vol_ { 0.0 };
};

/**
 * @brief Convert a lattice tag to a string.
 *
 * @param tag Lattice tag.
 * @return String corresponding to the given lattice tag.
 */
[[nodiscard]] std::string lattice_tag_to_string(const bravais_lattice::lattice_tag& tag);

/**
 * @brief Convert a string to a lattice tag.
 *
 * @param str String.
 * @return Lattice tag corresponding to the given certain string.
 */
[[nodiscard]] bravais_lattice::lattice_tag string_to_lattice_tag(const std::string& str);

/**
 * @brief Check 1d lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_1d_lattice_parameters(const bravais_lattice::params& p);

/**
 * @brief Make linear lattice.
 *
 * @param a Lattice constant.
 * @return Linear lattice.
 */
[[nodiscard]] bravais_lattice make_linear_lattice(double a);

/**
 * @brief Check 2d lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_2d_lattice_parameters(const bravais_lattice::params& p);

/**
 * @brief Make square lattice.
 *
 * @param a Lattice constant.
 * @return Square lattice.
 */
[[nodiscard]] bravais_lattice make_square_lattice(double a);

/**
 * @brief Make 2d hexagonal lattice.
 *
 * @param a Lattice constant.
 * @return 2d hexagonal lattice.
 */
[[nodiscard]] bravais_lattice make_hexagonal_lattice(double a);

/**
 * @brief Make rectangular lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @return Rectangular lattice.
 */
[[nodiscard]] bravais_lattice make_rectangular_lattice(double a, double b);

/**
 * @brief Make rectangular centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @return Rectangular centered lattice.
 */
[[nodiscard]] bravais_lattice make_rectangular_centered_lattice(double a, double b);

/**
 * @brief Make oblique 2d lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param gamma Angle between a-b.
 * @return Oblique 2d lattice.
 */
[[nodiscard]] bravais_lattice make_oblique_lattice(double a, double b, double gamma);

/**
 * @brief Check 3d lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_3d_lattice_parameters(const bravais_lattice::params& p);

/**
 * @brief Make cubic lattice.
 *
 * @param a Lattice constant.
 * @return Cubic lattice.
 */
[[nodiscard]] bravais_lattice make_cubic_lattice(double a);

/**
 * @brief Make FCC lattice.
 *
 * @param a Lattice constant.
 * @return FCC lattice.
 */
[[nodiscard]] bravais_lattice make_fcc_lattice(double a);

/**
 * @brief Make BCC lattice.
 *
 * @param a Lattice constant.
 * @return BCC lattice.
 */
[[nodiscard]] bravais_lattice make_bcc_lattice(double a);

/**
 * @brief Make hexagonal lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Hexagonal lattice.
 */
[[nodiscard]] bravais_lattice make_hexagonal_lattice(double a, double c);

/**
 * @brief Make rhombohedral lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Rhombohedral lattice.
 */
[[nodiscard]] bravais_lattice make_rhombohedral_lattice(double a, double c);

/**
 * @brief Make tetragonal lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Tetragonal lattice.
 */
[[nodiscard]] bravais_lattice make_tetragonal_lattice(double a, double c);

/**
 * @brief Make tetragonal body-centered lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Tetragonal body-centered lattice.
 */
[[nodiscard]] bravais_lattice make_tetragonal_bc_lattice(double a, double c);

/**
 * @brief Make orthorhombic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic lattice.
 */
[[nodiscard]] bravais_lattice make_orthorhombic_lattice(double a, double b, double c);

/**
 * @brief Make orthorombic body-centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic body-centered lattice.
 */
[[nodiscard]] bravais_lattice make_orthorhombic_bc_lattice(double a, double b, double c);

/**
 * @brief Make orthorombic face-centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic face-centered lattice.
 */
[[nodiscard]] bravais_lattice make_orthorhombic_fc_lattice(double a, double b, double c);

/**
 * @brief Make orthorombic base-centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic base-centered lattice.
 */
[[nodiscard]] bravais_lattice make_orthorhombic_base_centered_lattice(double a, double b, double c);

/**
 * @brief Make monoclinic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @param beta Angle between a-c.
 * @return Monoclinic lattice.
 */
[[nodiscard]] bravais_lattice make_monoclinic_lattice(double a, double b, double c, double beta);

/**
 * @brief Make monoclinic base-centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @param beta Angle between a-c.
 * @return Monoclinic base-centered lattice.
 */
[[nodiscard]] bravais_lattice make_monoclinic_base_centered_lattice(double a, double b, double c, double beta);

/**
 * @brief Make triclinic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @param alpha Angle between b-c.
 * @param beta Angle between a-c.
 * @param gamma Angle between a-b.
 * @return Triclinic lattice.
 */
[[nodiscard]] bravais_lattice make_triclinic_lattice(
    double a, double b, double c, double alpha, double beta, double gamma);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
