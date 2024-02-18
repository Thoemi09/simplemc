/**
 * @file utils.hpp
 * @brief Utility functions for simplemc-accs.
 */

#ifndef SIMPLEMC_ACCS_UTILS_HPP
#define SIMPLEMC_ACCS_UTILS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <cassert>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace simplemc::accs {

/**
 * @brief Enumerate the different algorithms how to calculate means, variances, etc.
 */
enum class varalg { standard, welford };

/**
 * @brief Make an Eigen::Array with NaNs.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @param size Size of storage.
 * @return Eigen::Array with NaNs.
 */
template <double_or_complex T>
Eigen::ArrayX<T> make_nans(long size) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        return Eigen::ArrayX<T>::Constant(size, nan);
    } else {
        return Eigen::ArrayX<T>::Constant(size, std::complex<double> { nan, nan });
    }
}

/**
 * @brief Calculate the sample mean.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to calculate the mean.
 * @param data Accumulated data.
 * @param count Number of accumulated values.
 * @param shift Constant shift applied to the accumulated values.
 * @return Sample mean.
 */
template <double_or_complex T, varalg A = varalg::standard>
Eigen::ArrayX<T> mean(const Eigen::ArrayX<T>& data, std::uint64_t count, const Eigen::ArrayX<T>& shift) {
    assert(data.size() == shift.size());
    if (count == 0) {
        return make_nans<T>(data.size());
    }
    if constexpr (A == varalg::standard) {
        return data / static_cast<T>(count) + shift;
    } else {
        return data + shift;
    }
}

/**
 * @brief Calculate the sample variance.
 *
 * @tparam A Algorithm used to calculate the variance.
 * @param data Accumulated data.
 * @param data2 Accumulated squared data.
 * @param count Number of accumulated values.
 * @return Sample variance.
 */
template <varalg A = varalg::standard>
Eigen::ArrayX<double> variance(
    [[maybe_unused]] const Eigen::ArrayX<double>& data, const Eigen::ArrayX<double>& data2, std::uint64_t count) {
    if (count <= 1) {
        return make_nans<double>(data.size());
    }
    const auto cd = static_cast<double>(count);
    if constexpr (A == varalg::standard) {
        return (data2 - data * data / cd) / (cd - 1);
    } else {
        return data2 / (cd - 1);
    }
}

} // namespace simplemc::accs

#endif // SIMPLEMC_ACCS_UTILS_HPP
