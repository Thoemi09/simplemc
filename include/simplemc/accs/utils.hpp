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
 * @brief Make an Eigen::ArrayX with NaNs.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @param size Size of storage.
 * @return Eigen::ArrayX with NaNs.
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
 * @brief Make an Eigen::ArrayXX with NaNs.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @param size Size of storage.
 * @return Eigen::ArrayXX with NaNs.
 */
template <double_or_complex T>
Eigen::ArrayXX<T> make_nans(long rows, long cols) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        return Eigen::ArrayXX<T>::Constant(rows, cols, nan);
    } else {
        return Eigen::ArrayXX<T>::Constant(rows, cols, std::complex<double> { nan, nan });
    }
}

/**
 * @brief Calculate the sample mean of a random vector.
 *
 * @details We have to correct for the constant shift vector.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to calculate the mean.
 * @param mdata Accumulated mean data.
 * @param count Number of accumulated values.
 * @param shift Constant shift vector applied to the accumulated values.
 * @return Sample mean.
 */
template <double_or_complex T, varalg A = varalg::standard>
Eigen::ArrayX<T> mean(const Eigen::ArrayX<T>& mdata, std::uint64_t count, const Eigen::ArrayX<T>& shift) {
    assert(data.size() == shift.size());
    if (count == 0) {
        return make_nans<T>(mdata.size());
    }
    if constexpr (A == varalg::standard) {
        return mdata / static_cast<double>(count) + shift;
    } else {
        return mdata + shift;
    }
}

/**
 * @brief Calculate the diagonal elements of the sample covariance or cross-covariance matrix of
 * two random vectors.
 *
 * @tparam A Algorithm used to calculate the variance.
 * @param mdata1 Accumulated mean data #1.
 * @param mdata2 Accumulated mean data #2.
 * @param vdata Accumulated covariance/cross-covariance data.
 * @param count Number of accumulated values.
 * @return Diagonal of the sample covariance or cross-covariance matrix.
 */
template <varalg A = varalg::standard>
Eigen::ArrayX<double> diag_covariance([[maybe_unused]] const Eigen::ArrayX<double>& mdata1,
    [[maybe_unused]] const Eigen::ArrayX<double>& mdata2, const Eigen::ArrayX<double>& vdata, std::uint64_t count) {
    assert(mdata1.size() == mdata2.size());
    assert(mdata1.size() == vdata.size());
    if (count <= 1) {
        return make_nans<double>(mdata1.size());
    }
    const auto cd = static_cast<double>(count);
    if constexpr (A == varalg::standard) {
        return (vdata - mdata1 * mdata2 / cd) / (cd - 1);
    } else {
        return vdata / (cd - 1);
    }
}

} // namespace simplemc::accs

#endif // SIMPLEMC_ACCS_UTILS_HPP
