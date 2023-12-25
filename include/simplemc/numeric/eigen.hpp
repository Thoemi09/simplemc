/**
 * @file eigen.hpp
 * @brief Include Eigen library with some useful extensions.
 */

#ifndef SIMPLEMC_NUMERIC_EIGEN_HPP
#define SIMPLEMC_NUMERIC_EIGEN_HPP

#include <Eigen/Dense>

#include <limits>
#include <span>

namespace simplemc {

/**
 * @brief Templated, static, double, N-dimensional vector type.
 *
 * @tparam N Number of dimensions.
 */
template <std::size_t N>
struct vector {
    using type = Eigen::Matrix<double, N, 1>;
};

/**
 * @brief Templated, static, double N x N matrix type.
 *
 * @tparam N Number of dimensions.
 */
template <std::size_t N>
struct matrix {
    using type = Eigen::Matrix<double, N, N>;
};

/**
 * @brief Create a span from an Eigen::PlainObjectBase object.
 *
 * @tparam Derived Derived type.
 * @param t Eigen::PlainObjectBase object.
 * @return Span of the object.
 */
template <typename Derived>
inline auto make_span(const Eigen::PlainObjectBase<Derived>& t) {
    return std::span<std::remove_reference_t<decltype(t(0, 0))>>(&t(0, 0), static_cast<std::size_t>(t.size()));
}

template <typename Derived>
inline auto make_span(Eigen::PlainObjectBase<Derived>& t) {
    return std::span<std::remove_reference_t<decltype(t(0, 0))>>(&t(0, 0), static_cast<std::size_t>(t.size()));
}

/**
 * @brief Checks if an Eigen::MatrixBase object is finite.
 *
 * @tparam Derived Derived type.
 * @param t Eigen::MatrixBase object.
 * @return `true`, if any element is inf or nan.
 */
template <typename Derived>
inline bool isfinite(const Eigen::MatrixBase<Derived>& t) {
    return t.array().isfinite().any();
}

/**
 * @brief Absolute difference between two Eigen::Matrix using the 2-norm for vectors
 * and the Frobenius norm for matrices.
 *
 * @tparam Derived Derived type.
 * @param t1 Vector/Matrix #1.
 * @param t2 Vector/Matrix #2.
 * @return 2-norm/Frobenius norm of their difference.
 */
template <typename Derived>
inline double abs_diff(const Eigen::MatrixBase<Derived>& t1, const Eigen::MatrixBase<Derived>& t2) {
    return (t1 - t2).norm();
}

/**
 * @brief Relative difference between two Eigen::Matrix using the 2-norm for vectors
 * and the Frobenius norm for matrices.
 *
 * @tparam Derived Derived type.
 * @param t1 Vector #1.
 * @param t2 Vector #2.
 * @return Relative difference of their norms.
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
 * @brief Transform polar to cartesian coordinates.
 *
 * @details It performs the following transformations:
 *
 * - 1D: (x) -> (x)
 * - 2D: (r, phi) -> (x, y)
 * - 3D: (r, theta, phi) -> (x, y, z)
 *
 * @param vec Vector with spherical/polar coordinates (r, theta, phi)/(r, phi)/(x).
 * @return Vector with cartesian vector (x, y, z)/(x, y)/(x).
 */
inline typename vector<1>::type polar_to_cartesian(const typename vector<1>::type& vec) {
    return vec;
}

/// See simplemc::polar_to_cartesian.
inline typename vector<2>::type polar_to_cartesian(const typename vector<2>::type& vec) {
    typename vector<2>::type res;
    res << vec[0] * std::cos(vec[1]), vec[0] * std::sin(vec[1]);
    return res;
}

/// See simplemc::polar_to_cartesian.
inline typename vector<3>::type polar_to_cartesian(const typename vector<3>::type& vec) {
    typename vector<3>::type res;
    const auto sin2 = std::sin(vec[2]);
    res << vec[0] * std::cos(vec[1]) * sin2, vec[0] * std::cos(vec[2]), vec[0] * std::sin(vec[1]) * sin2;
    return res;
}

/**
 * @brief Transform cartesian to polar coordinates.
 *
 * @details It performs the following transformations:
 *
 * - 1D: (x) -> (x)
 * - 2D: (x, y) -> (r, phi)
 * - 3D: (x, y, z) -> (r, theta, phi)
 *
 * @param vec Vector with cartesian coordinates (x, y, z)/(x, y)/(x).
 * @return Vector with spherical/polar coordinates (r, theta, phi)/(r, phi)/(x).
 */
inline typename vector<1>::type cartesian_to_polar(const typename vector<1>::type& vec) {
    return vec;
}

/// See simplemc::cartesian_to_polar.
inline typename vector<2>::type cartesian_to_polar(const typename vector<2>::type& vec) {
    typename vector<2>::type res;
    res << vec.norm(), std::atan2(vec[1], vec[0]);
    return res;
}

/// See simplemc::cartesian_to_polar.
inline typename vector<3>::type cartesian_to_polar(const typename vector<3>::type& vec) {
    typename vector<3>::type res;
    const auto norm = vec.norm();
    res << norm, std::acos(vec[2] / norm), std::atan2(vec[1], vec[0]);
    return res;
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_EIGEN_HPP
