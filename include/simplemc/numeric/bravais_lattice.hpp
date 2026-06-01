/**
 * @file
 * @brief General Bravais lattice class and corresponding utilities.
 */

#ifndef SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
#define SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-lattices
 * @{
 */

/**
 * @brief Tags for the different Bravais lattices.
 *
 * @details The following lattices are supported with corresponding factory functions (bc:
 * body-centered, fc: face-centered):
 * - 1D:
 *   - `linear_1d` (simplemc::make_linear_lattice)
 * - 2D:
 *   - `square_2d` (simplemc::make_square_lattice
 *   - `hexagonal_2d` (simplemc::make_hexagonal_lattice)
 *   - `rectangular_2d` (simplemc::make_rectangular_lattice)
 *   - `rectangular_centered_2d` (simplemc::make_rectangular_centered_lattice)
 *   - `oblique_2d` (simplemc::make_oblique_lattice)
 * - 3D:
 *   - `cubic_3d` (simplemc::make_cubic_lattice)
 *   - `fcc_3d` (simplemc::make_fcc_lattice)
 *   - `bcc_3d` (simplemc::make_bcc_lattice)
 *   - `hexagonal_3d` (@ref simplemc::make_hexagonal_lattice(double, double)
 *   "simplemc::make_hexagonal_lattice")
 *   - `rhombohedral_3d` (simplemc::make_rhombohedral_lattice)
 *   - `tetragonal_3d` (simplemc::make_tetragonal_lattice)
 *   - `tetragonal_bc_3d` (simplemc::make_tetragonal_bc_lattice)
 *   - `orthorhombic_3d` (simplemc::make_orthorhombic_lattice)
 *   - `orthorhombic_fc_3d` (simplemc::make_orthorhombic_fc_lattice)
 *   - `orthorhombic_bc_3d` (simplemc::make_orthorhombic_bc_lattice)
 *   - `orthorhombic_base_centered_3d` (simplemc::make_orthorhombic_base_centered_lattice)
 *   - `monoclinic_3d` (simplemc::make_monoclinic_lattice)
 *   - `monoclinic_base_centered_3d` (simplemc::make_monoclinic_base_centered_lattice)
 *   - `triclinic_3d` (simplemc::make_triclinic_lattice)
 *
 * The default tag is `unspecified`.
 */
enum class lattice_tag {
    // functions (like check_lattice_tag) depend on the order
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
 * @param tag simplemc::lattice_tag.
 * @return `std::string` corresponding to the given lattice tag.
 */
[[nodiscard]] std::string lattice_tag_to_string(const lattice_tag& tag);

/**
 * @brief Convert a string to a lattice tag.
 *
 * @param str `std::string_view` object.
 * @return simplemc::lattice_tag corresponding to the given string.
 */
[[nodiscard]] lattice_tag string_to_lattice_tag(std::string_view str);

/**
 * @brief Parameters that specify the conventional unit cell of a Bravais lattice.
 *
 * @details We assume that the lattice vectors for the conventional unit cell are given by
 * - \f$ \{ \mathbf{a} \} \f$ in 1D,
 * - \f$ \{ \mathbf{a}, \mathbf{b} \} \f$ in 2D and
 * - \f$ \{ \mathbf{a}, \mathbf{b}, \mathbf{c} \} \f$ in 3D.
 *
 * Depending on the dimensionality of the lattice, not all parameters will be set.
 *
 * @note It is recommended to let the factory functions in @ref simplemc-numeric-lattices-1d,
 * @ref simplemc-numeric-lattices-2d and @ref simplemc-numeric-lattices-3d set the correct parameters.
 */
struct lattice_parameters {
    /// Lattice constant \f$ a \f$ in direction \f$ \mathbf{a} \f$.
    double a { 0.0 };

    /// Lattice constant \f$ b \f$ in direction \f$ \mathbf{b} \f$ (only in 2D and 3D).
    double b { 0.0 };

    /// Lattice constant \f$ c \f$ in direction \f$ \mathbf{c} \f$ (only in 3D).
    double c { 0.0 };

    /// Angle \f$ \alpha \f$ between \f$ \mathbf{b} \f$ and \f$ \mathbf{c} \f$ (only in 3D).
    double alpha { 0.0 };

    /// Angle \f$ \beta \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{c} \f$ (only in 3D)
    double beta { 0.0 };

    /// Angle \f$ \gamma \f$ between \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$ (only in 2D and 3D).
    double gamma { 0.0 };

    /// simplemc::lattice_tag specifying the type of Bravais lattice.
    lattice_tag tag { lattice_tag::unspecified };
};

/**
 * @brief Check that (lattice) basis vectors are linearly independent.
 *
 * @details It throws a simplemc::simplemc_exception, if \f$ |\det(\mathbf{A})| < \epsilon \f$, where
 * \f$ \mathbf{A} = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \f$ is a matrix containing the basis
 * vectors in its columns and \f$ \epsilon \f$ is some threshold.
 *
 * @tparam N Number of dimensions.
 * @param A Matrix \f$ \mathbf{A} \f$ with the basis vectors in its columns.
 * @param eps Threshold \f$ \epsilon \f$.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
void check_basis_vectors(const Eigen::Matrix<double, N, N>& A, double eps = 1e-14) {
    if (is_matrix_singular(A, eps)) {
        throw simplemc_exception("Basis vectors are linearly dependent");
    }
}

/**
 * @brief Calculate the volume of the cell spanned by given vectors.
 *
 * @details Let \f$ \mathbf{C} = \big(\mathbf{c}_1 \cdots \mathbf{c}_N \big) \f$ be the matrix
 * containing the vectors in its columns. Then the volume of the N-dimensional cell is given by \f$
 * |\det(\mathbf{C})| \f$.
 *
 * @tparam N Number of dimensions.
 * @param C Matrix \f$ \mathbf{C} \f$ with the vectors in its columns.
 * @return Volume spanned by the given vectors.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
[[nodiscard]] double calculate_cell_volume(const Eigen::Matrix<double, N, N>& C) {
    return std::abs(C.determinant());
}

/**
 * @brief Calculate the reciprocal lattice basis vectors \f$ \{ \mathbf{b}_1, \dots, \mathbf{b}_N \}
 * \f$.
 *
 * @details Let \f$ \mathbf{A} = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \f$ be the matrix
 * containing the basis vectors of a Bravais lattice. The corresponding reciprocal basis vectors, \f$
 * \mathbf{B} = \big(\mathbf{b}_1 \cdots \mathbf{b}_N \big) \f$, satisfy the equation \f$ \mathbf{A}
 * \mathbf{B}^T = 2 \pi \mathbf{I} \f$ such that \f$ \mathbf{B} = 2 \pi \left( \mathbf{A}^{-1}
 * \right)^T \f$.
 *
 * @tparam N Number of dimensions.
 * @param A Matrix \f$ \mathbf{A} \f$ with the real space basis vectors in its columns.
 * @return Matrix \f$ \mathbf{B} \f$ with the reciprocal basis vectors in its columns.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
[[nodiscard]] auto calculate_reciprocal_lattice(const Eigen::Matrix<double, N, N>& A) {
    using std::numbers::pi;
    return A.transpose().fullPivLu().inverse() * 2.0 * pi;
}

/**
 * @brief Calculate the vertices of the cell spanned by given vectors.
 *
 * @details Let \f$ \mathbf{C} = \big(\mathbf{c}_1 \cdots \mathbf{c}_N \big) \f$ be the matrix
 * containing the vectors in its columns. Then the vertices \f$ \mathbf{V} \f$ of the cell spanned by
 * those vectors are:
 * - \f$ \mathbf{V} = \left( \mathbf{0}, \; \mathbf{c}_1 \right) \f$ in 1D,
 * - \f$ \mathbf{V} = \left( \mathbf{0}, \; \mathbf{c}_1, \; \mathbf{c}_1 + \mathbf{c}_2, \;
 * \mathbf{c}_2 \right) \f$ in 2D and
 * - \f$ \mathbf{V} = \left( \mathbf{0}, \; \mathbf{c}_1, \; \mathbf{c}_1 + \mathbf{c}_2, \;
 * \mathbf{c}_2, \; \mathbf{c}_3, \; \mathbf{c}_1 + \mathbf{c}_3, \; \mathbf{c}_1 + \mathbf{c}_2 +
 * \mathbf{c}_3, \; \mathbf{c}_2 + \mathbf{c}_3 \right) \f$ in 3D.
 *
 * @tparam N Number of dimensions.
 * @param C Matrix \f$ \mathbf{C} \f$ with the vectors in its columns.
 * @return Matrix \f$ \mathbf{V} \f$ with the corresponding vertices in its columns.
 */
template <int N>
    requires(N == 1 || N == 2 || N == 3)
[[nodiscard]] auto calculate_cell_vertices(const Eigen::Matrix<double, N, N>& C) {
    if constexpr (N == 1) {
        using mat_t = Eigen::Matrix<double, 1, 2>;
        mat_t res = mat_t::Zero();
        res(0, 1) = C(0, 0);
        return res;
    } else if constexpr (N == 2) {
        using mat_t = Eigen::Matrix<double, 2, 4>;
        mat_t res = mat_t::Zero();
        res.col(1) = C.col(0);
        res.col(2) = C.col(0) + C.col(1);
        res.col(3) = C.col(1);
        return res;
    } else {
        using mat_t = Eigen::Matrix<double, 3, 8>;
        mat_t res = mat_t::Zero();
        res.col(1) = C.col(0);
        res.col(2) = C.col(0) + C.col(1);
        res.col(3) = C.col(1);
        res.col(4) = C.col(2);
        res.col(5) = C.col(2) + C.col(0);
        res.col(6) = C.col(2) + C.col(0) + C.col(1);
        res.col(7) = C.col(2) + C.col(1);
        return res;
    }
}

/**
 * @brief Class representing a Bravais lattice in 1D, 2D and 3D.
 *
 * @details The Bravais lattice and its reciprocal lattice are defined in
 * @ref simplemc-numeric-lattices. This class simply stores
 * - the matrix \f$ \mathbf{A} = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \f$ containing the basis
 * vectors of the lattice,
 * - the matrix \f$ \mathbf{B} = \big(\mathbf{b}_1 \cdots \mathbf{b}_N \big) \f$ containing the basis
 * vectors of the reciprocal lattice,
 * - the volume of the real space unit cell and
 * - the volume of the reciprocal space unit cell.
 *
 * @note **simplemc** provides several factory functions to create Bravais lattices in 1D, 2D and 3D
 * (see @ref simplemc-numeric-lattices-1d, @ref simplemc-numeric-lattices-2d and
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
     * @brief Get the number of dimensions of the Bravais lattice.
     *
     * @return Number of dimensions.
     */
    [[nodiscard]] static constexpr int dim() noexcept { return N; };

    /**
     * @brief Default constructor leaves the Bravais lattice unspecified and uninitialized.
     */
    bravais_lattice() = default;

    /**
     * @brief Construct a Bravais lattice with the given basis vectors.
     *
     * @details It simply forwards the basis vector to reset().
     *
     * @param A Matrix \f$ \mathbf{A} = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \f$ with the real
     * space basis vectors in its columns.
     */
    bravais_lattice(const matrix_type& A) { reset(A); }

    /**
     * @brief Reset the Bravais lattice using the given basis vectors.
     *
     * @details The basis vectors are first checked with simplemc::check_basis_vectors.
     *
     * @param A Matrix \f$ \mathbf{A} = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \f$ with the real
     * space basis vectors in its columns.
     */
    void reset(const matrix_type& A) {
        check_basis_vectors(A);
        A_ = A;
        B_ = calculate_reciprocal_lattice(A_);
        volA_ = calculate_cell_volume(A_);
        volB_ = calculate_cell_volume(B_);
    }

    /**
     * @brief Get the real space basis vectors of the lattice.
     *
     * @return Matrix \f$ \mathbf{A} = \big(\mathbf{a}_1 \cdots \mathbf{a}_N \big) \f$ with the real
     * space basis vectors in its columns.
     */
    [[nodiscard]] const matrix_type& real_lattice_vectors() const noexcept { return A_; }

    /**
     * @brief Get the unit cell volume spanned by the real space lattice vectors.
     *
     * @return Unit cell volume in real space, i.e. \f$ |\det(\mathbf{A})| \f$.
     */
    [[nodiscard]] double real_cell_volume() const noexcept { return volA_; }

    /**
     * @brief Get the reciprocal space lattice vectors.
     *
     * @return Matrix \f$ \mathbf{B} = \big(\mathbf{b}_1 \cdots \mathbf{b}_N \big) \f$ with the
     * reciprocal basis vectors in its columns.
     */
    [[nodiscard]] const matrix_type& reciprocal_lattice_vectors() const noexcept { return B_; }

    /**
     * @brief Get the unit cell volume spanned by the reciprocal space lattice vectors.
     *
     * @return Unit cell volume in reciprocal space, i.e. \f$ |\det(\mathbf{B})| \f$.
     */
    [[nodiscard]] double reciprocal_cell_volume() const noexcept { return volB_; }

private:
    matrix_type A_;
    matrix_type B_;
    double volA_ { 0.0 };
    double volB_ { 0.0 };
};

/// `bravais_lattice<N>` is reconstructed from its real-space basis matrix; B/volumes are derived.
template <class S, int N>
    requires serializer<std::remove_cvref_t<S>>
void simplemc_save(S&& s, const bravais_lattice<N>& l) {
    s.save_at("A", l.real_lattice_vectors());
}

template <class S, int N>
    requires deserializer<std::remove_cvref_t<S>>
void simplemc_load(S&& s, bravais_lattice<N>& l) {
    typename bravais_lattice<N>::matrix_type A;
    s.load_at("A", A);
    l.reset(A);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_BRAVAIS_LATTICE_HPP
