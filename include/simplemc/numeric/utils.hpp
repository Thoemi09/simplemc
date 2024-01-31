/**
 * @file utils.hpp
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
 * @brief Check if an arithmetic value is finite.
 *
 * @details Simply calls std::isfinite.
 *
 * @tparam T Value type.
 * @param t Arithmetic value.
 * @return True if the value is finite.
 */
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] inline constexpr bool isfinite(T t) {
    return std::isfinite(t);
}

/**
 * @brief Check if a complex number is finite.
 *
 * @details Simply calls std::isfinite on the real and imaginary part.
 *
 * @tparam T Value type.
 * @param t std::complex<T> object.
 * @return True if the real and the imaginary part are finite.
 */
template <typename T>
[[nodiscard]] inline constexpr bool isfinite(const std::complex<T>& t) {
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
    requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
[[nodiscard]] inline constexpr auto abs_diff(T1 t1, T2 t2) {
    return std::abs(t1 - t2);
}

/**
 * @brief Absolute difference between two complex numbers.
 *
 * @tparam T Value type of std::complex.
 * @param t1 std::complex<T> #1.
 * @param t2 std::complex<T> #2.
 * @return Modulus of their difference.
 */
template <typename T>
[[nodiscard]] inline constexpr auto abs_diff(const std::complex<T>& t1, const std::complex<T>& t2) {
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
 * @brief Relative difference between two complex numbers.
 *
 * @tparam T Value type of std::complex.
 * @param t1 std::complex<T> #1.
 * @param t2 std::complex<T> #2.
 * @return Relative difference.
 */
template <typename T>
[[nodiscard]] inline constexpr auto rel_diff(const std::complex<T>& t1, const std::complex<T>& t2) {
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
 * @brief Map the given value to its principle interval `(lower_bound, upper_bound]`
 * excluding the lower bound and including the upper bound.
 *
 * @details The difference `upper_bound - lower_bound` will be added to the given value
 * until it lies within `(lower_bound, upper_bound]`.
 *
 * @param val Value to be mapped.
 * @param lower_bound Lower bound of the interval.
 * @param upper_bound Upper bound of the interval.
 * @return Value mapped to its principle interval.
 */
[[nodiscard]] double map_to_interval(double val, double lower_bound, double upper_bound);

/**
 * @brief Map the given value to its principle interval `[lower_bound, upper_bound)`
 * excluding the upper bound and including the lower bound.
 *
 * @details The difference `upper_bound - lower_bound` will be added to the given value
 * until it lies within `[lower_bound, upper_bound)`.
 *
 * @param val Value to be mapped.
 * @param lower_bound Lower bound of the interval.
 * @param upper_bound Upper bound of the interval.
 * @return Value mapped to its principle interval.
 */
[[nodiscard]] double map_to_interval_lb(double val, double lower_bound, double upper_bound);

/**
 * @brief Check if a value lies within certain bounds, i.e. in the interval
 * `(lower_bound + min_diff, upper_bound - min_diff)`.
 *
 * @param val Value to be checked.
 * @param lower_bound Lower bound.
 * @param upper_bound Upper bound.
 * @param min_diff Minimum distance to bounds.
 * @return True if value lies within `(lower_bound + min_diff, upper_bound - min_diff)`.
 */
[[nodiscard]] inline constexpr bool within_bounds(double val, double lower_bound, double upper_bound, double min_diff) {
    return val > lower_bound + min_diff && val < upper_bound - min_diff && isfinite(val);
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_UTILS_HPP
