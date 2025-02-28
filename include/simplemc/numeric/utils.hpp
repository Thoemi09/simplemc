/**
 * @file
 * @brief Utility functions for simplemc-numeric.
 */

#ifndef SIMPLEMC_NUMERIC_UTILS_HPP
#define SIMPLEMC_NUMERIC_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-utils
 * @{
 */

/**
 * @brief Check if an arithmetic value is finite.
 *
 * @details Simply calls `std::isfinite`.
 *
 * @tparam T Arithmetic type.
 * @param t Arithmetic value.
 * @return True if the value is finite.
 */
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] inline constexpr bool isfinite(T t) {
    return std::isfinite(t);
}

/**
 * @brief Check if a complex value is finite.
 *
 * @details Simply calls `std::isfinite` on the real and imaginary part.
 *
 * @tparam T Value type of `std::complex`.
 * @param z Complex value.
 * @return True if the real and the imaginary parts are finite.
 */
template <typename T>
[[nodiscard]] inline constexpr bool isfinite(const std::complex<T>& z) {
    return isfinite(z.real()) && isfinite(z.imag());
}

/**
 * @brief Absolute difference between two arithmetic values.
 *
 * @tparam T1 Arithmetic type.
 * @tparam T2 Arithmetic type.
 * @param t1 Arithmetic value #1.
 * @param t2 Arithmetic value #2.
 * @return Absolute value of their difference, i.e. \f$ | t_1 - t_2 | \f$.
 */
template <typename T1, typename T2>
    requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
[[nodiscard]] inline constexpr auto abs_diff(T1 t1, T2 t2) {
    return std::abs(t1 - t2);
}

/**
 * @brief Absolute difference between two complex values.
 *
 * @tparam T Value type of `std::complex`.
 * @param z1 Complex value #1.
 * @param z2 Complex value #2.
 * @return Modulus of their difference, i.e. \f$ | z_1 - z_2 | \f$.
 */
template <typename T>
[[nodiscard]] inline constexpr auto abs_diff(const std::complex<T>& z1, const std::complex<T>& z2) {
    return std::abs(z1 - z2);
}

/**
 * @brief Relative difference between two arithmetic values.
 *
 * @details We define the relative difference as \f$ | t_1 - t_2 | / \min\{ |t_1|, |t_2| \} \f$.
 *
 * In case that \f$ t_1 = 0 \f$ or \f$ t_2 = 0 \f$, we set the value to the very small number
 * `std::numeric_limits<double>::min()`, respectively.
 *
 * @tparam T1 Arithmetic type.
 * @tparam T2 Arithmetic type.
 * @param t1 Arithmetic value #1.
 * @param t2 Arithmetic value #2.
 * @return Their absolute difference divided by the smaller absolute value of the two values.
 */
template <typename T1, typename T2>
    requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
[[nodiscard]] inline constexpr auto rel_diff(T1 t1, T2 t2) {
    double a = t1;
    double b = t2;
    if (t1 == 0) {
        a = std::numeric_limits<double>::min();
    }
    if (t2 == 0) {
        b = std::numeric_limits<double>::min();
    }
    return std::max(std::abs((a - b) / a), std::abs((a - b) / b));
}

/**
 * @brief Relative difference between two complex values.
 *
 * @details We define the relative difference as \f$ | z_1 - z_2 | / \min\{ |z_1|, |z_2| \} \f$.
 *
 * In case that \f$ |z_1| = 0 \f$ or \f$ |z_2| = 0 \f$, we set the value to the very small real number
 * `std::numeric_limits<double>::min()`, respectively.
 *
 * @tparam T Value type of `std::complex`.
 * @param z1 Complex value #1.
 * @param z2 Complex value #2.
 * @return Their absolute difference divided by the smaller modulus of the two values.
 */
template <typename T>
[[nodiscard]] inline constexpr auto rel_diff(const std::complex<T>& z1, const std::complex<T>& z2) {
    double a = std::abs(z1);
    double b = std::abs(z2);
    if (a == 0) {
        a = std::numeric_limits<double>::min();
    }
    if (b == 0) {
        b = std::numeric_limits<double>::min();
    }
    return std::max(std::abs((z1 - z2) / a), std::abs((z1 - z2) / b));
}

/**
 * @brief Map a given value \f$ x \f$ to the interval \f$ (a, b] \f$ excluding the lower bound and
 * including the upper bound.
 *
 * @details The difference \f$ b - a \f$ will be added to/subtracted from \f$ x \f$ until it lies
 * within \f$ (a, b] \f$.
 *
 * @param x Value \f$ x \f$ to be mapped.
 * @param a Lower bound of the interval.
 * @param b Upper bound of the interval.
 * @return Mapped value \f$ \in (a, b] \f$.
 */
[[nodiscard]] double map_to_interval(double x, double a, double b);

/**
 * @brief Map a given value \f$ x \f$ to the interval \f$ [a, b) \f$ excluding the upper bound and
 * including the lower bound.
 *
 * @details The difference \f$ b - a \f$ will be added to/subtracted from \f$ x \f$ until it lies
 * within \f$ [a, b) \f$.
 *
 * @param x Value \f$ x \f$ to be mapped.
 * @param a Lower bound of the interval.
 * @param b Upper bound of the interval.
 * @return Mapped value \f$ \in [a, b) \f$.
 */
[[nodiscard]] double map_to_interval_lb(double x, double a, double b);

/**
 * @brief Check if a value \f$ x \f$ lies within certain bounds, i.e. if \f$ a + \epsilon \leq x \leq
 * b - \epsilon \f$.
 *
 * @param x Value \f$ x \f$ to be checked.
 * @param a Lower bound \f$ a \f$.
 * @param b Upper bound \f$ b \f$.
 * @param eps Minimum distance \f$ \epsilon \f$ to bounds.
 * @return True if \f$ x \in [a + \epsilon, b - \epsilon] \f$.
 */
[[nodiscard]] inline constexpr bool within_bounds(double x, double a, double b, double eps) {
    return x >= a + eps && x <= b - eps && isfinite(x);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_UTILS_HPP
