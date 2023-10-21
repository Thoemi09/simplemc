/**
 * @file utils.hpp
 * @brief Utility functions for simplemc-numeric.
 */

#ifndef SIMPLEMC_NUMERICS_UTILS_HPP
#define SIMPLEMC_NUMERICS_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>

namespace simplemc {

/**
 * @brief Checks if an arithmetic value is finite. Simply calls std::isfinite.
 *
 * @tparam T Value type.
 * @param t Arithmetic value.
 * @return True if value is finite.
 */
template <typename T>
inline constexpr bool isfinite(T t) {
    return std::isfinite(t);
}

/**
 * @brief Checks if a complex number is finite.
 *
 * @tparam T Value type.
 * @param t std::complex<T> object.
 * @return True if real and imaginary part are finite.
 */
template <typename T>
inline constexpr bool isfinite(const std::complex<T>& t) {
    return isfinite(t.real()) && isfinite(t.imag());
}

/**
 * @brief Absolute difference between two arithmetic values.
 *
 * @tparam T1 Type of value #1.
 * @tparam T2 Type of value #2.
 * @param t1 Value #1.
 * @param t2 Value #2.
 * @return Absolute difference.
 */
template <typename T1, typename T2>
inline constexpr auto abs_diff(T1 t1, T2 t2) {
    return std::abs(t1 - t2);
}

/**
 * @brief Absolute difference between two complex numbers.
 *
 * @tparam T Value type.
 * @param t1 std::complex<T> #1.
 * @param t2 std::complex<T> #2.
 * @return Modulus of their difference.
 */
template <typename T>
inline constexpr auto abs_diff(const std::complex<T>& t1, const std::complex<T>& t2) {
    return std::abs(t1 - t2);
}

/**
 * @brief Relative difference between two arithmetic values.
 *
 * @tparam T1 Type of value #1.
 * @tparam T2 Type of value #2.
 * @param t1 Value #1.
 * @param t2 Value #2.
 * @return Relative difference.
 */
template <typename T1, typename T2>
inline constexpr auto rel_diff(T1 t1, T2 t2) {
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
 * @brief Relative difference between two complex numbers.
 *
 * @tparam T Value type.
 * @param t1 std::complex<T> #1.
 * @param t2 std::complex<T> #2.
 * @return Relative difference.
 */
template <typename T>
inline constexpr auto rel_diff(const std::complex<T>& t1, const std::complex<T>& t2) {
    double a = std::abs(t1);
    double b = std::abs(t2);
    if (a == 0) {
        a = std::numeric_limits<double>::min();
    }
    if (b == 0) {
        b = std::numeric_limits<double>::min();
    }
    return std::max(std::abs((t1 - t2) / a), std::abs((t1 - t2) / b));
}

/**
 * @brief Calculates the index of a subset such that the current position lies in the
 * middle of it.
 *
 * @details Let I = {0, 1, ..., n-1}. Given the current position l in the set, we want
 * to find the position j of the subset {j, j+1, ..., j+m-1} such that l lies in the
 * middle of this subset (insofar as this is possible considering the boundaries, i.e. j
 * >= 0 and j <= n-1). If m is even, the smaller possible j value is chosen, e.g. if m = 2,
 * then j = l - 1.
 *
 * @tparam T Integral type.
 * @param l Current position.
 * @param n Length/Size of the array/grid.
 * @param m Size of the subinterval.
 * @return Index j such that l lies in the middle of {j, j+1, ..., j+m-1}.
 */
template <std::integral T>
inline constexpr auto index_of_subset(T l, T n, T m) {
    assert(l >= 0 && l < n);
    assert(m > 0 && m <= n);
    return std::clamp(l - m / 2, 0, std::max(0, n - m));
}

/**
 * @brief Map the given value to its principle interval (lower_bound, upper_bound].
 *
 * @param val Value to be mapped.
 * @param lower_bound Lower bound of interval.
 * @param upper_bound Upper bound of interval.
 * @return Value mapped to its principle interval.
 */
double map_to_interval(double val, double lower_bound, double upper_bound);

/**
 * @brief Map the given value to its principle interval [lower_bound, upper_bound).
 *
 * @param val Value to be mapped.
 * @param lower_bound Lower bound of interval.
 * @param upper_bound Upper bound of interval.
 * @return Value mapped to its principle interval.
 */
double map_to_interval_lb(double val, double lower_bound, double upper_bound);

/**
 * @brief Check if a value lies within certain bounds, i.e. in the interval
 * (lower_bound + min_diff, upper_bound - min_diff).
 *
 * @param val Value to be checked.
 * @param lower_bound Lower bound.
 * @param upper_bound Upper bound.
 * @param min_diff Minimum distance to bounds.
 * @return True, if value lies within (lower_bound + min_diff, upper_bound - min_diff).
 */
inline constexpr bool within_bounds(double val, double lower_bound, double upper_bound, double min_diff) {
    return val > lower_bound + min_diff && val < upper_bound - min_diff && isfinite(val);
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERICS_UTILS_HPP
