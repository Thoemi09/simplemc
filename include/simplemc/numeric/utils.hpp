/**
 * @file
 * @brief Utility functions for simplemc-numeric.
 */

#ifndef SIMPLEMC_NUMERIC_UTILS_HPP
#define SIMPLEMC_NUMERIC_UTILS_HPP

#include <algorithm>
#include <cassert>
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
 * @details It simply calls `std::isfinite`.
 *
 * @tparam T Arithmetic type.
 * @param t Arithmetic value \f$ t \f$.
 * @return True if the value is finite.
 */
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr bool isfinite(T t) {
    return std::isfinite(t);
}

/**
 * @brief Check if a complex value is finite.
 *
 * @details It simply calls simplemc::isfinite on the real and imaginary part.
 *
 * @tparam T Value type of `std::complex`.
 * @param z Complex value \f$ z \f$.
 * @return True if the real and the imaginary parts are finite.
 */
template <typename T>
[[nodiscard]] constexpr bool isfinite(const std::complex<T>& z) {
    return isfinite(z.real()) && isfinite(z.imag());
}

/**
 * @brief Absolute difference between two arithmetic values.
 *
 * @tparam T1 Arithmetic type.
 * @tparam T2 Arithmetic type.
 * @param t1 Arithmetic value \f$ t_1 \f$.
 * @param t2 Arithmetic value \f$ t_1 \f$.
 * @return Absolute value of their difference, i.e. \f$ | t_1 - t_2 | \f$.
 */
template <typename T1, typename T2>
    requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
[[nodiscard]] constexpr auto abs_diff(T1 t1, T2 t2) {
    return std::abs(t1 - t2);
}

/**
 * @brief Absolute difference between two complex values.
 *
 * @tparam T Value type of `std::complex`.
 * @param z1 Complex value \f$ z_1 \f$.
 * @param z2 Complex value \f$ z_1 \f$.
 * @return Modulus of their difference, i.e. \f$ | z_1 - z_2 | \f$.
 */
template <typename T>
[[nodiscard]] constexpr auto abs_diff(const std::complex<T>& z1, const std::complex<T>& z2) {
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
 * @param t1 Arithmetic value \f$ t_1 \f$.
 * @param t2 Arithmetic value \f$ t_2 \f$.
 * @return Their absolute difference divided by the smaller absolute value of the two values.
 */
template <typename T1, typename T2>
    requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
[[nodiscard]] constexpr auto rel_diff(T1 t1, T2 t2) {
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
 * @param z1 Complex value \f$ z_1 \f$.
 * @param z2 Complex value \f$ z_2 \f$.
 * @return Their absolute difference divided by the smaller modulus of the two values.
 */
template <typename T>
[[nodiscard]] constexpr auto rel_diff(const std::complex<T>& z1, const std::complex<T>& z2) {
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
 * @brief Map a given value \f$ x \f$ to the interval \f$ (a, b] \f$.
 *
 * @details The difference \f$ b - a \f$ will be added to/subtracted from \f$ x \f$ until it lies
 * within \f$ (a, b] \f$.
 *
 * The upper bound \f$ b \f$ is included, while the lower bound \f$ a \f$ is excluded.
 *
 * @param x Value \f$ x \f$ to be mapped.
 * @param a Lower bound of the interval.
 * @param b Upper bound of the interval.
 * @return Mapped value \f$ \in (a, b] \f$.
 */
[[nodiscard]] inline constexpr double map_to_interval(double x, double a, double b) {
    assert(b > a);
    const auto len = b - a;
    const auto mid = (b + a) * 0.5;
    const auto res = x > mid ? std::fmod(x - a, len) + a : std::fmod(x - b, len) + b;
    return (res == a ? b : res);
}

/**
 * @brief Map a given value \f$ x \f$ to the interval \f$ [a, b) \f$.
 *
 * @details The difference \f$ b - a \f$ will be added to/subtracted from \f$ x \f$ until it lies
 * within \f$ [a, b) \f$.
 *
 * The lower bound \f$ a \f$ is included, while the upper bound \f$ b \f$ is excluded.
 *
 * @param x Value \f$ x \f$ to be mapped.
 * @param a Lower bound of the interval.
 * @param b Upper bound of the interval.
 * @return Mapped value \f$ \in [a, b) \f$.
 */
[[nodiscard]] inline constexpr double map_to_interval_lb(double x, double a, double b) {
    assert(b > a);
    const auto len = b - a;
    const auto mid = (b + a) * 0.5;
    const auto res = x > mid ? std::fmod(x - a, len) + a : std::fmod(x - b, len) + b;
    return (res == b ? a : res);
}

/**
 * @brief Check if a value \f$ x \f$ lies within certain bounds.
 *
 * @details It checks if \f$ a + \epsilon \leq x \leq b - \epsilon \f$.
 *
 * @param x Value \f$ x \f$ to be checked.
 * @param a Lower bound \f$ a \f$.
 * @param b Upper bound \f$ b \f$.
 * @param eps Minimum distance \f$ \epsilon \f$ to bounds.
 * @return True if \f$ x \in [a + \epsilon, b - \epsilon] \f$.
 */
[[nodiscard]] inline constexpr bool within_bounds(double x, double a, double b, double eps = 0.0) {
    return x >= a + eps && x <= b - eps && isfinite(x);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_UTILS_HPP
