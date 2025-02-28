/**
 * @file
 * @brief Includes the Eigen library and adds some useful extensions.
 */

#ifndef SIMPLEMC_NUMERIC_EIGEN_HPP
#define SIMPLEMC_NUMERIC_EIGEN_HPP

#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-utils
 * @{
 */

/**
 * @brief A concept that checks if an integer value can be used as a static extent for Eigen matrices.
 *
 * @details The given integer value must be either `Eigen::Dynamic` or a positive integer.
 *
 * @tparam M Static extent.
 */
template <int M>
concept static_extent = (M == Eigen::Dynamic || M > 0);

/**
 * @brief A concept that checks if a type is an `Eigen::Matrix<T, M, N>` type.
 *
 * @details `T` has to be either `double` or `std::complex<double>`, and `M` and `N` have to be both
 * valid simplemc::static_extent integers.
 *
 * @tparam M Type to check.
 */
template <typename M>
concept eigen_matrix = requires {
    typename M::Scalar;
    M::RowsAtCompileTime;
    M::ColsAtCompileTime;
    requires std::is_same_v<M, Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime>>;
    requires double_or_complex<typename M::Scalar>;
    requires static_extent<M::RowsAtCompileTime>;
    requires static_extent<M::ColsAtCompileTime>;
};

/**
 * @brief A concept that checks if a type is an `Eigen::Matrix<double, M, N>` type.
 *
 * @details `M` and `N` have to be both valid simplemc::static_extent integers.
 *
 * @tparam M Type to check.
 */
template <typename M>
concept eigen_matrix_dbl = (eigen_matrix<M> && std::is_same_v<typename M::Scalar, double>);

/**
 * @brief A concept that checks if a type is an `Eigen::Matrix<std::complex<double>, M, N>` type.
 *
 * @details `M` and `N` have to be both valid simplemc::static_extent integers.
 *
 * @tparam M Type to check.
 */
template <typename M>
concept eigen_matrix_cplx = (eigen_matrix<M> && std::is_same_v<typename M::Scalar, std::complex<double>>);

/**
 * @brief A concept that checks if a type is an `Eigen::Matrix<T, M, 1>` type.
 *
 * @details `T` has to be either `double` or `std::complex<double>`, and `M` has to be a valid
 * simplemc::static_extent.
 *
 * @tparam V Type to check.
 */
template <typename V>
concept eigen_vector = (eigen_matrix<V> && V::ColsAtCompileTime == 1);

/**
 * @brief A concept that checks if a type is an `Eigen::Matrix<double, M, 1>` type.
 *
 * @details `M` has to be a valid simplemc::static_extent.
 *
 * @tparam V Type to check.
 */
template <typename V>
concept eigen_vector_dbl = (eigen_vector<V> && std::is_same_v<typename V::Scalar, double>);

/**
 * @brief A concept that checks if a type is an `Eigen::Matrix<std::complex<double>, M, 1>` type.
 *
 * @details `M` has to be a valid simplemc::static_extent.
 *
 * @tparam V Type to check.
 */
template <typename V>
concept eigen_vector_cplx = (eigen_vector<V> && std::is_same_v<typename V::Scalar, std::complex<double>>);

/**
 * @brief Make an `Eigen::Matrix<T, M, 1>` with NaNs.
 *
 * @note For static sized objects, the `size` parameter is ignored.
 *
 * @tparam V simplemc::eigen_vector type.
 * @param size Size of the vector.
 * @return `Eigen::Matrix<T, M, 1>` with NaNs.
 */
template <typename V>
    requires(eigen_vector_dbl<V> || eigen_vector_cplx<V>)
V nans_vector(long size = 1) {
    size = (V::RowsAtCompileTime == Eigen::Dynamic ? size : static_cast<long>(V::RowsAtCompileTime));
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<typename V::Scalar, double>) {
        return V::Constant(size, nan);
    } else {
        return V::Constant(size, std::complex<double> { nan, nan });
    }
}

/**
 * @brief Make an `Eigen::Matrix<T, M, N>` with NaNs.
 *
 * @note For static sized objects, the `rows` and `cols` parameters are ignored.
 *
 * @tparam M simplemc::eigen_matrix type.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return `Eigen::Matrix<T, M, N>` with NaNs.
 */
template <eigen_matrix M>
M nans_matrix(long rows = 1, long cols = 1) {
    rows = (M::RowsAtCompileTime == Eigen::Dynamic ? rows : static_cast<long>(M::RowsAtCompileTime));
    cols = (M::ColsAtCompileTime == Eigen::Dynamic ? cols : static_cast<long>(M::ColsAtCompileTime));
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<typename M::Scalar, double>) {
        return M::Constant(rows, cols, nan);
    } else {
        return M::Constant(rows, cols, std::complex<double> { nan, nan });
    }
}

/**
 * @brief Create a `std::span` from an `Eigen::PlainObjectBase` object.
 *
 * @tparam Derived Derived Eigen type.
 * @param t `Eigen::PlainObjectBase` object.
 * @return `std::span` of the underlying data.
 */
template <typename Derived>
inline auto make_span(const Eigen::PlainObjectBase<Derived>& t) {
    return std::span<std::remove_reference_t<decltype(t(0, 0))>>(&t(0, 0), static_cast<std::size_t>(t.size()));
}

/// Overload of simplemc::make_span for non-const `Eigen::PlainObjectBase` objects.
template <typename Derived>
inline auto make_span(Eigen::PlainObjectBase<Derived>& t) {
    return std::span<std::remove_reference_t<decltype(t(0, 0))>>(&t(0, 0), static_cast<std::size_t>(t.size()));
}

/**
 * @brief Check if an `Eigen::MatrixBase` object is finite.
 *
 * @tparam Derived Derived Eigen type.
 * @param t `Eigen::MatrixBase` object.
 * @return True if all elements of the object are finite.
 */
template <typename Derived>
inline bool isfinite(const Eigen::MatrixBase<Derived>& t) {
    return t.array().isFinite().all();
}

/**
 * @brief Absolute difference between two `Eigen::Matrix` objects using the 2-norm for vectors and the
 * Frobenius norm for matrices.
 *
 * @details It calculates
 * - \f$ \lVert t_1 - t_2 \rVert_2 \f$ for vectors and
 * - \f$ \lVert t_1 - t_2 \rVert_F \f$ for matrices.
 *
 * @tparam Derived1 Derived Eigen type.
 * @tparam Derived2 Dervied Eigen type.
 * @param t1 `Eigen::MatrixBase` object #1.
 * @param t2 `Eigen::MatrixBase` object #2.
 * @return 2-norm/Frobenius norm of their difference.
 */
template <typename Derived1, typename Derived2>
inline double abs_diff(const Eigen::MatrixBase<Derived1>& t1, const Eigen::MatrixBase<Derived2>& t2) {
    return (t1 - t2).norm();
}

/**
 * @brief Relative difference between two `Eigen::Matrix` objects using the 2-norm for vectors and the
 * Frobenius norm for matrices.
 *
 * @details It calculates
 * - \f$ \lVert t_1 - t_2 \rVert_2 / \min\{ \lVert t_1 \rVert_2, \lVert t_2 \rVert_2 \}\f$ for vectors
 * and
 * - \f$ \lVert t_1 - t_2 \rVert_F / \min\{ \lVert t_1 \rVert_F, \lVert t_2 \rVert_F \}\f$ for
 * matrices.
 *
 * In case that \f$ \lVert t_1 \rVert_2 = 0 \f$ or \f$ \lVert t_2 \rVert_2 = 0 \f$, we set the value
 * to the very small real number `std::numeric_limits<double>::min()`, respectively.
 *
 * @tparam Derived1 Derived Eigen type.
 * @tparam Derived2 Dervied Eigen type.
 * @param t1 `Eigen::MatrixBase` object #1.
 * @param t2 `Eigen::MatrixBase` object #2.
 * @return Their absolute difference divided by the norm of the object with the smaller norm.
 */
template <typename Derived1, typename Derived2>
inline double rel_diff(const Eigen::MatrixBase<Derived1>& t1, const Eigen::MatrixBase<Derived2>& t2) {
    double n = (t1 - t2).norm();
    double a = t1.norm();
    double b = t2.norm();
    if (a == 0) {
        a = std::numeric_limits<double>::min();
    }
    if (b == 0) {
        b = std::numeric_limits<double>::min();
    }
    return std::max(n / a, n / b);
}

/**
 * @brief Transform polar to cartesian coordinates in 1D.
 *
 * @details In 1D, this is simply the identity transformation, i.e. \f$ (x) \to (x) \f$.
 *
 * @param vec `Eigen::Matrix<double, 1, 1>` with spherical/polar coordinates \f$ (x) \f$.
 * @return `Eigen::Matrix<double, 1, 1>` with cartesian coordinates \f$ (x) \f$.
 */
inline auto polar_to_cartesian(const Eigen::Matrix<double, 1, 1>& vec) {
    return vec;
}

/**
 * @brief Transform polar to cartesian coordinates in 2D.
 *
 * @details The following transformations are performed \f$ (r, \phi) \to (x, y) \f$:
 *
 * - \f$ x = r \cos(\phi) \f$ and
 * - \f$ y = r \sin(\phi) \f$.
 *
 * @param vec `Eigen::Vector2d` with spherical/polar coordinates \f$ (r, \phi) \f$.
 * @return `Eigen::Vector2d` with cartesian coordinates \f$ (x, y) \f$.
 */
inline auto polar_to_cartesian(const Eigen::Vector2d& vec) {
    Eigen::Vector2d res;
    res << vec[0] * std::cos(vec[1]), vec[0] * std::sin(vec[1]);
    return res;
}

/**
 * @brief Transform spherical to cartesian coordinates in 3D.
 *
 * @details The following transformations are performed \f$ (r, \theta, \phi) \to (x, y, z) \f$:
 *
 * - \f$ x = r \cos(\phi) \sin(\theta) \f$,
 * - \f$ y = r \sin(\phi) \sin(\theta) \f$ and
 * - \f$ z = r \cos(\theta) \f$.
 *
 * @param vec `Eigen::Vector3d` with spherical/polar coordinates \f$ (r, \theta, \phi) \f$.
 * @return `Eigen::Vector3d` with cartesian coordinates \f$ (x, y, z) \f$.
 */
inline auto polar_to_cartesian(const Eigen::Vector3d& vec) {
    Eigen::Vector3d res;
    const auto sin_theta = std::sin(vec[1]);
    res << vec[0] * std::cos(vec[2]) * sin_theta, vec[0] * std::sin(vec[2]) * sin_theta, vec[0] * std::cos(vec[1]);
    return res;
}

/**
 * @brief Transform cartesian to polar coordinates in 1D.
 *
 * @details In 1D, this is simply the identity transformation, i.e. \f$ (x) \to (x) \f$.
 *
 * @param vec `Eigen::Matrix<double, 1, 1>` with cartesian coordinates \f$ (x) \f$.
 * @return `Eigen::Matrix<double, 1, 1>` with spherical/polar coordinates \f$ (x) \f$.
 */
inline auto cartesian_to_polar(const Eigen::Matrix<double, 1, 1>& vec) {
    return vec;
}

/**
 * @brief Transform cartesian to polar coordinates in 2D.
 *
 * @details The following transformations are performed \f$ (x, y) \to (r, \phi) \f$:
 *
 * - \f$ r = \sqrt{x^2 + y^2} \f$ and
 * - \f$ \phi = \mathrm{atan2}(y, x) \f$.
 *
 * @param vec `Eigen::Vector2d` with cartesian coordinates \f$ (x, y) \f$.
 * @return `Eigen::Vector2d` with spherical/polar coordinates \f$ (r, \phi) \f$.
 */
inline auto cartesian_to_polar(const Eigen::Vector2d& vec) {
    Eigen::Vector2d res;
    res << vec.norm(), std::atan2(vec[1], vec[0]);
    return res;
}

/**
 * @brief Transform cartesian to polar coordinates in 3D.
 *
 * @details The following transformations are performed \f$ (x, y, z) \to (r, \theta, \phi) \f$:
 *
 * - \f$ r = \sqrt{x^2 + y^2 + z^2} \f$,
 * - \f$ \theta = \arccos\left(\frac{z}{r}\right) \f$ and
 * - \f$ \phi = \mathrm{atan2}(y, x) \f$.
 *
 * @param vec `Eigen::Vector3d` with cartesian coordinates \f$ (x, y, z) \f$.
 * @return `Eigen::Vector3d` with spherical/polar coordinates \f$ (r, \theta, \phi) \f$.
 */
inline auto cartesian_to_polar(const Eigen::Vector3d& vec) {
    Eigen::Vector3d res;
    const auto norm = vec.norm();
    res << norm, std::acos(vec[2] / norm), std::atan2(vec[1], vec[0]);
    return res;
}

/**
 * @brief Calculate the angle between two vectors.
 *
 * @details The angle between two vectors \f$ \mathbf{u} \f$ and \f$ \mathbf{v} \f$ is given by
 * \f[
 *   \Theta = \angle(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u}
 *   \rVert \lVert \mathbf{v} \rVert} \; .
 * \f]
 *
 * @tparam Derived1 Derived Eigen type.
 * @tparam Derived2 Dervied Eigen type.
 * @param u Vector \f$ \mathbf{u} \f$.
 * @param v Vector \f$ \mathbf{v} \f$.
 * @return Angle \f$ \Theta \f$ between the two vectors.
 */
template <typename Derived1, typename Derived2>
double angle(const Eigen::MatrixBase<Derived1>& u, const Eigen::MatrixBase<Derived2>& v) {
    return std::acos(u.dot(v) / u.norm() / v.norm());
}

/**
 * @brief Check if a matrix is singluar.
 *
 * @details A matrix \f$ \mathbf{M} \f$ is considered to be singular if \f$ |\det(\mathbf{M})| \leq
 * \epsilon \f$, where \f$ \epsilon \f$ is some threshold.
 *
 * @tparam Derived Derived Eigen type.
 * @param mat Matrix \f$ \mathbf{M} \f$ to be checked.
 * @param eps Threshold \f$ \epsilon \f$.
 * @return True, if the matrix is considered to be singular.
 */
template <typename Derived>
bool is_matrix_singular(const Eigen::MatrixBase<Derived>& mat, double eps = 1e-14) {
    assert(mat.rows() == mat.cols());
    return std::abs(mat.determinant()) <= eps;
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_EIGEN_HPP
