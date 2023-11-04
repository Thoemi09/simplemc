/**
 * @file bravais_lattice.hpp
 * @brief Bravais lattices in 1d, 2d and 3d.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP

#include <simplemc/numeric/eigen.hpp>

#include <string>

namespace simplemc {

/**
 * @brief Bravais lattice class.
 *
 * @details A lattice is defined by d d-dimensional lattice vectors. One lattice point is
 * in the origin and all others are obtained by translating the origin using the lattice
 * vectors.
 *
 * The lattice can be in 1, 2 or 3 dimensions. Other dimensions are not supported.
 *
 * It is assumed that the lattice vectors are linearly independent and that the lattice
 * parameters are correctly set. It is recommended to use the make_*_lattice functions
 * instead of manually creating a lattice.
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
     * @brief Tag for bravais lattice.
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
     * @brief Parameters specifying a bravais lattice.
     *
     * @details The following parameters are stored:
     *
     *     - a: Lattice constant in direction #1.
     *     - b: Lattice constant in direction #2 (only in 2d and 3d).
     *     - c: Lattice constant in direction #3 (only in 3d).
     *     - alpha: Angle between b and c (only in 3d).
     *     - beta: Angle between a and c (only in 3d).
     *     - gamma: Angle between a and b (only in 2d and 3d).
     *     - tag: Tag for the lattice.
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
     * @brief Check dimensions of lattice vector matrix. Throws an exception if dimensions are
     * not supported, i.e. only support for 1x1, 2x2 and 3x3.
     *
     * @param mat Matrix with lattice vectors in columns.
     */
    static void check_lattice_vector_dims(const matrix_type& mat);

    /**
     * @brief Calculate volume of a given cell.
     *
     * @param mat Matrix with lattice vectors in columns.
     * @return Volume spanned by lattice vectors.
     */
    static double calculate_cell_volume(const matrix_type& mat);

    /**
     * @brief Calculate reciprocal lattice vectors w.r.t. to a given lattice.
     *
     * @param mat Matrix with lattice vectors in columns.
     * @return Matrix with reciprocal lattice vectors in columns.
     */
    static matrix_type calculate_reciprocal_lattice(const matrix_type& mat);

    /**
     * @brief Calculate vertices of the cell spanned by given lattice vectors.
     *
     * @param mat Matrix with lattice vectors in columns.
     * @return Matrix with vertices in columns.
     */
    static matrix_type calculate_cell_vertices(const matrix_type& mat);

    /**
     * @brief Calculate the transformations matrix of the linear map which maps the cell spanned
     * by the given lattice vectors to the unit line/square/cube.
     *
     * @param mat Matrix with lattice vectors in columns.
     * @return Transformation matrix.
     */
    static matrix_type calculate_transformation_matrix(const matrix_type& mat);

    /**
     * @brief Default constructor.
     */
    bravais_lattice() = default;

    /**
     * @brief Construct a bravais lattice with lattice vectors and parameters.
     *
     * @param mat Matrix with lattice vectors in columns.
     * @param p Bravais lattice parameters.
     */
    bravais_lattice(matrix_type mat, const params& p);

    /**
     * @brief Number of dimensions.
     */
    auto dim() const { return real_lat_.rows(); };

    /**
     * @brief Set lattice vectors.
     *
     * @param mat Matrix with lattice vectors in columns.
     */
    void set_lattice_vectors(matrix_type mat);

    /**
     * @brief Set lattice parameters.
     *
     * @param p Bravais lattice parameters.
     */
    void set_lattice_parameters(const params& p);

    /**
     * @brief Get bravais lattice parameters.
     *
     * @return Bravais lattice parameters.
     */
    [[nodiscard]] const params& get_lattice_parameters() const { return params_; }

    /**
     * @brief Get real space lattice vectors.
     *
     * @return Real space lattice vectors.
     */
    [[nodiscard]] const matrix_type& real_lattice_vectors() const { return real_lat_; }

    /**
     * @brief Get unit cell volume spanned by real space lattice vectors.
     *
     * @return Real space unit cell volume.
     */
    [[nodiscard]] double real_cell_volume() const { return real_vol_; }

    /**
     * @brief Get reciprocal space lattice vectors.
     *
     * @return Reciprocal space lattice vectors.
     */
    [[nodiscard]] const matrix_type& reciprocal_lattice_vectors() const { return rec_lat_; }

    /**
     * @brief Get unit cell volume spanned by reciprocal space lattice vectors.
     *
     * @return Reciprocal space unit cell volume.
     */
    [[nodiscard]] double reciprocal_cell_volume() const { return rec_vol_; }

protected:
    params params_;
    matrix_type real_lat_;
    matrix_type rec_lat_;
    double real_vol_ { 0.0 };
    double rec_vol_ { 0.0 };
};

/**
 * @brief Convert lattice tag to string.
 *
 * @param tag Lattice tag.
 * @return String corresponding to a certain lattice tag.
 */
std::string lattice_tag_to_string(const bravais_lattice::lattice_tag& tag);

/**
 * @brief Convert string to lattice tag.
 *
 * @param str String.
 * @return Lattice tag corresponding to a certain string.
 */
bravais_lattice::lattice_tag string_to_lattice_tag(const std::string& str);

/**
 * @brief Check 1d lattice parameters.
 *
 * @param p Lattice parameters.
 */
void check_1d_lattice_parameters(const bravais_lattice::params& p);

/**
 * @brief Make linear lattice vectors.
 *
 * @param a Lattice constant.
 * @return Linear lattice.
 */
bravais_lattice make_linear_lattice(double a);

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
bravais_lattice make_square_lattice(double a);

/**
 * @brief Make 2d hexagonal lattice.
 *
 * @param a Lattice constant.
 * @return 2d hexagonal lattice.
 */
bravais_lattice make_hexagonal_lattice(double a);

/**
 * @brief Make rectangular lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @return Rectangular lattice.
 */
bravais_lattice make_rectangular_lattice(double a, double b);

/**
 * @brief Make rectangular centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @return Rectangular centered lattice.
 */
bravais_lattice make_rectangular_centered_lattice(double a, double b);

/**
 * @brief Make oblique 2d lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param gamma Angle between a-b.
 * @return Oblique 2d lattice.
 */
bravais_lattice make_oblique_lattice(double a, double b, double gamma);

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
bravais_lattice make_cubic_lattice(double a);

/**
 * @brief Make FCC lattice.
 *
 * @param a Lattice constant.
 * @return FCC lattice.
 */
bravais_lattice make_fcc_lattice(double a);

/**
 * @brief Make BCC lattice.
 *
 * @param a Lattice constant.
 * @return BCC lattice.
 */
bravais_lattice make_bcc_lattice(double a);

/**
 * @brief Make hexagonal lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Hexagonal lattice.
 */
bravais_lattice make_hexagonal_lattice(double a, double c);

/**
 * @brief Make rhombohedral lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Rhombohedral lattice.
 */
bravais_lattice make_rhombohedral_lattice(double a, double c);

/**
 * @brief Make tetragonal lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Tetragonal lattice.
 */
bravais_lattice make_tetragonal_lattice(double a, double c);

/**
 * @brief Make tetragonal body centered lattice.
 *
 * @param a Lattice constant.
 * @param c Lattice constant.
 * @return Tetragonal body centered lattice.
 */
bravais_lattice make_tetragonal_bc_lattice(double a, double c);

/**
 * @brief Make orthorhombic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic lattice.
 */
bravais_lattice make_orthorhombic_lattice(double a, double b, double c);

/**
 * @brief Make orthorombic body centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic body centered lattice.
 */
bravais_lattice make_orthorhombic_bc_lattice(double a, double b, double c);

/**
 * @brief Make orthorombic face centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic face centered lattice.
 */
bravais_lattice make_orthorhombic_fc_lattice(double a, double b, double c);

/**
 * @brief Make orthorombic base centered lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @return Orthorhombic base centered lattice.
 */
bravais_lattice make_orthorhombic_base_centered_lattice(double a, double b, double c);

/**
 * @brief Make orthorhombic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @param beta Angle between a-c.
 * @return Monoclinic lattice.
 */
bravais_lattice make_monoclinic_lattice(double a, double b, double c, double beta);

/**
 * @brief Make orthorhombic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @param beta Angle between a-c.
 * @return Monoclinic base centered lattice.
 */
bravais_lattice make_monoclinic_base_centered_lattice(double a, double b, double c, double beta);

/**
 * @brief Make orthorhombic lattice.
 *
 * @param a Lattice constant.
 * @param b Lattice constant.
 * @param c Lattice constant.
 * @param alpha Angle between b-c.
 * @param beta Angle between a-c.
 * @param gamma Angle between a-b.
 * @return Triclinic lattice.
 */
bravais_lattice make_triclinic_lattice(double a, double b, double c, double alpha, double beta, double gamma);

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
