/**
 * @file
 * @brief General Bravais lattice class and corresponding utilities.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP

#include <simplemc/numeric/eigen.hpp>

#include <numbers>
#include <string>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices
 * @{
 */

/**
 * @brief Tags for the different Bravais lattices.
 *
 * @details The following lattices are implemented:
 * - 1D: linear
 * - 2D: square, hexagonal, rectangular, centered rectangular, oblique
 * - 3D: cubic, FCC, BCC, hexagonal, rhombohedral, tetragonal, tetragonal body-centered, orthorhombic,
 * orthorhombic face-centered, orthorhombic body-centered, orthorhombic base-centered, monoclinic,
 * monoclinic base-centered, triclinic
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
 * @brief Convert a lattice tag to a string.
 *
 * @param tag Lattice tag.
 * @return String corresponding to the given lattice tag.
 */
[[nodiscard]] std::string lattice_tag_to_string(const lattice_tag& tag);

/**
 * @brief Convert a string to a lattice tag.
 *
 * @param str String.
 * @return Lattice tag corresponding to the given certain string.
 */
[[nodiscard]] lattice_tag string_to_lattice_tag(const std::string& str);

/**
 * @brief Parameters for specifying a Bravais lattice.
 *
 * @details The following parameters are stored:
 *
 * - \f$ a \f$: Lattice constant in direction #1.
 * - \f$ b \f$: Lattice constant in direction #2 (only in 2D and 3D).
 * - \f$ c \f$: Lattice constant in direction #3 (only in 3D).
 * - \f$ \alpha \f$: Angle between \f$ \mathbf{b} \f$ and \f$ \mathbf{c} \f$ (only in 3D).
 * - \f$ \beta \f$: Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$ (only in 3D).
 * - \f$ \gamma \f$: Angle between \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$ (only in 2D and 3D).
 * - `tag`: Tag specifying the type of Bravais lattice.
 */
struct lattice_parameters {
    double a { 0.0 };
    double b { 0.0 };
    double c { 0.0 };
    double alpha { 0.0 };
    double beta { 0.0 };
    double gamma { 0.0 };
    lattice_tag tag { lattice_tag::unspecified };
};

/**
 * @brief Calculate the volume of the cell spanned by the given lattice vectors.
 *
 * @tparam N Number of dimensions.
 * @param mat Matrix with the lattice vectors in its columns.
 * @return Volume spanned by the lattice vectors.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
[[nodiscard]] double calculate_cell_volume(const Eigen::Matrix<double, N, N>& mat) {
    return std::abs(mat.determinant());
}

/**
 * @brief Calculate the reciprocal lattice vectors w.r.t. to the given lattice vectors.
 *
 * @tparam N Number of dimensions.
 * @param mat Matrix with the lattice vectors in its columns.
 * @return Matrix with the corresponding reciprocal lattice vectors in its columns.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
[[nodiscard]] auto calculate_reciprocal_lattice(const Eigen::Matrix<double, N, N>& mat) {
    using std::numbers::pi;
    return mat.transpose().fullPivLu().inverse() * 2.0 * pi;
}

/**
 * @brief Calculate the vertices of the cell spanned by given lattice vectors.
 *
 * @details The following matrix sizes are returned:
 * - \f$ 1 \times 2 \f$ in 1D (2 vertices),
 * - \f$ 2 \times 4 \f$ in 2D (4 vertices) and
 * - \f$ 3 \times 8 \f$ in 3D (8 vertices).
 *
 * @tparam N Number of dimensions.
 * @param mat Matrix with the lattice vectors in its columns.
 * @return Matrix with the vertices in its columns.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
[[nodiscard]] auto calculate_cell_vertices(const Eigen::Matrix<double, N, N>& mat) {
    if constexpr (N == 1) {
        auto res = Eigen::Matrix<double, 1, 2>::Zero();
        res(0, 1) = mat(0, 0);
        return res;
    } else if constexpr (N == 2) {
        auto res = Eigen::Matrix<double, 2, 4>::Zero();
        res.col(1) = mat.col(0);
        res.col(2) = mat.col(0) + mat.col(1);
        res.col(3) = mat.col(1);
        return res;
    } else {
        auto res = Eigen::Matrix<double, 3, 8>::Zero();
        res.col(1) = mat.col(0);
        res.col(2) = mat.col(0) + mat.col(1);
        res.col(3) = mat.col(1);
        res.col(4) = mat.col(2);
        res.col(5) = mat.col(2) + mat.col(0);
        res.col(6) = mat.col(2) + mat.col(0) + mat.col(1);
        res.col(7) = mat.col(2) + mat.col(1);
        return res;
    }
}

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
 *
 * @tparam N Number of dimensions.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
class bravais_lattice {
public:
    /**
     * @brief Type of vector used.
     */
    using vector_type = Eigen::Vector<double, N>;

    /**
     * @brief Type of matrix used.
     */
    using matrix_type = Eigen::Matrix<double, N, N>;

    /**
     * @brief Lattice parameter type.
     */
    using param_type = lattice_parameters;

    /**
     * @brief Get the number of dimensions of the Bravais lattice.
     *
     * @return Number of dimensions.
     */
    [[nodiscard]] static constexpr int dim() { return N; };

    /**
     * @brief Default constructor leaves the Bravais lattice unspecified and uninitialized.
     */
    bravais_lattice() = default;

    /**
     * @brief Construct a Bravais lattice with lattice vectors and parameters.
     *
     * @warning No checks are performed to ensure that the given lattice vectors and parameters are
     * consistent.
     *
     * @param mat Matrix with the lattice vectors in its columns.
     * @param p Bravais lattice parameters.
     */
    bravais_lattice(const matrix_type& mat, const param_type& p) :
        param_(p),
        real_lat_(mat),
        rec_lat_(calculate_reciprocal_lattice(real_lat_)),
        real_vol_(calculate_cell_volume(real_lat_)),
        rec_vol_(calculate_cell_volume(rec_lat_)) {}

    /**
     * @brief Get the Bravais lattice parameters.
     *
     * @return Bravais lattice parameters.
     */
    [[nodiscard]] const param_type& param() const { return param_; }

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
    param_type param_;
    matrix_type real_lat_;
    matrix_type rec_lat_;
    double real_vol_ { 0.0 };
    double rec_vol_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
