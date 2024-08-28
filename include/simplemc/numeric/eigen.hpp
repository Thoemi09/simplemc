/**
 * @file
 * @brief Includes the Eigen library and adds some useful extensions.
 */

#ifndef SIMPLEMC_NUMERIC_EIGEN_HPP
#define SIMPLEMC_NUMERIC_EIGEN_HPP

#include <Eigen/Dense>

#include <algorithm>
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

// /**
//  * @brief Templated double vector type.
//  *
//  * @tparam N Number of dimensions.
//  */
// template <std::size_t N>
// struct vector {
//     using type = Eigen::Matrix<double, N, 1>;
// };

// /**
//  * @brief Templated double (square) matrix type.
//  *
//  * @tparam N Number of dimensions.
//  */
// template <std::size_t N>
// struct matrix {
//     using type = Eigen::Matrix<double, N, N>;
// };

/**
 * @brief Create a `std::span` from an `Eigen::PlainObjectBase` object.
 *
 * @tparam Derived Derived type.
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
 * @tparam Derived Derived type.
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
 * @tparam Derived Derived type.
 * @param t1 `Eigen::MatrixBase` object #1.
 * @param t2 `Eigen::MatrixBase` object #2.
 * @return 2-norm/Frobenius norm of their difference.
 */
template <typename Derived>
inline double abs_diff(const Eigen::MatrixBase<Derived>& t1, const Eigen::MatrixBase<Derived>& t2) {
    return (t1 - t2).norm();
}

/**
 * @brief Relative difference between two `Eigen::Matrix` objects using the 2-norm for vectors and the
 * Frobenius norm for matrices.
 *
 * @tparam Derived Derived type.
 * @param t1 `Eigen::MatrixBase` object #1.
 * @param t2 `Eigen::MatrixBase` object #2.
 * @return Their absolute difference divided by the norm of the object with the smaller norm.
 */
template <typename Derived>
inline double rel_diff(const Eigen::MatrixBase<Derived>& t1, const Eigen::MatrixBase<Derived>& t2) {
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
 * @param vec `Eigen::Matrix<double, 1, 1>` with spherical/polar coordinates.
 * @return `Eigen::Matrix<double, 1, 1>` with cartesian coordinates.
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
 * @param vec `Eigen::Vector2d` with spherical/polar coordinates.
 * @return `Eigen::Vector2d` with cartesian coordinates.
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
 * @param vec `Eigen::Vector3d` with spherical/polar coordinates.
 * @return `Eigen::Vector3d` with cartesian coordinates.
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
 * @param vec `Eigen::Matrix<double, 1, 1>` with cartesian coordinates.
 * @return `Eigen::Matrix<double, 1, 1>` with spherical/polar coordinates.
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
 * @param vec `Eigen::Vector2d` with cartesian coordinates.
 * @return `Eigen::Vector2d` with spherical/polar coordinates.
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
 * @param vec `Eigen::Vector3d` with cartesian coordinates.
 * @return `Eigen::Vector3d` with spherical/polar coordinates.
 */
inline auto cartesian_to_polar(const Eigen::Vector3d& vec) {
    Eigen::Vector3d res;
    const auto norm = vec.norm();
    res << norm, std::acos(vec[2] / norm), std::atan2(vec[1], vec[0]);
    return res;
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_EIGEN_HPP
