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
 * @brief Calculate the sample variance or covariance.
 *
 * @details If the given accumulated mean data storages are the same, then the sample variance is
 * calculated, otherwise the sample covariance.
 *
 * @tparam A Algorithm used to calculate the variance.
 * @param mdata1 Accumulated mean data #1.
 * @param mdata2 Accumulated mean data #2.
 * @param vdata Accumulated variance/covariance data.
 * @param count Number of accumulated values.
 * @return Sample variance or covariance.
 */
template <varalg A = varalg::standard>
Eigen::ArrayX<double> variance([[maybe_unused]] const Eigen::ArrayX<double>& mdata1,
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
