/**
 * @file
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
 * @addtogroup simplemc-accs
 */

/**
 * @brief Enumerate the different algorithms how to calculate means, variances, etc.
 */
enum class varalg { standard, welford };

/**
 * @brief Make an Eigen::VectorX with NaNs.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @param size Size of the vector.
 * @return Eigen::VectorX with NaNs.
 */
template <double_or_complex T>
Eigen::VectorX<T> make_nans(long size) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        return Eigen::VectorX<T>::Constant(size, nan);
    } else {
        return Eigen::VectorX<T>::Constant(size, std::complex<double> { nan, nan });
    }
}

/**
 * @brief Make an Eigen::MatrixX with NaNs.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Eigen::MatrixX with NaNs.
 */
template <double_or_complex T>
Eigen::MatrixX<T> make_nans(long rows, long cols) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        return Eigen::MatrixX<T>::Constant(rows, cols, nan);
    } else {
        return Eigen::MatrixX<T>::Constant(rows, cols, std::complex<double> { nan, nan });
    }
}

/**
 * @brief Calculate the sample mean of a random vector.
 *
 * @details We have to correct for the constant shift vector.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to accumulate the data.
 * @param mdata Accumulated mean data.
 * @param count Number of accumulated values.
 * @param shift Constant shift vector applied to the accumulated values.
 * @return Sample mean.
 */
template <double_or_complex T, varalg A = varalg::standard>
Eigen::VectorX<T> mean(const Eigen::VectorX<T>& mdata, std::uint64_t count, const Eigen::VectorX<T>& shift) {
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
 * @brief Calculate the diagonal elements of the sample (cross-)covariance matrix of two random vectors.
 *
 * @tparam A Algorithm used to accumulate the data.
 * @param mdata1 Accumulated mean data of random vector #1.
 * @param mdata2 Accumulated mean data of random vector #2.
 * @param cdata Accumulated (cross-)covariance data.
 * @param count Number of accumulated values.
 * @return Diagonal of the sample (cross-)covariance matrix.
 */
template <varalg A = varalg::standard>
Eigen::VectorX<double> diag_covariance([[maybe_unused]] const Eigen::VectorX<double>& mdata1,
    [[maybe_unused]] const Eigen::VectorX<double>& mdata2, const Eigen::VectorX<double>& cdata, std::uint64_t count) {
    assert(mdata1.size() == mdata2.size());
    assert(mdata1.size() == cdata.size());
    if (count <= 1) {
        return make_nans<double>(mdata1.size());
    }
    const auto cd = static_cast<double>(count);
    if constexpr (A == varalg::standard) {
        return (cdata - mdata1.cwiseProduct(mdata2) / cd) / (cd - 1);
    } else {
        return cdata / (cd - 1);
    }
}

/**
 * @brief Calculate the full sample (cross-)covariance matrix of two random vectors.
 *
 * @tparam A Algorithm used to accumulate the data.
 * @param mdata1 Accumulated mean data of random vector #1.
 * @param mdata2 Accumulated mean data of random vector #2.
 * @param cdata Accumulated (cross-)covariance data.
 * @param count Number of accumulated values.
 * @return Sample (cross-)covariance matrix.
 */
template <varalg A = varalg::standard>
Eigen::MatrixX<double> covariance([[maybe_unused]] const Eigen::VectorX<double>& mdata1,
    [[maybe_unused]] const Eigen::VectorX<double>& mdata2, const Eigen::MatrixX<double>& cdata, std::uint64_t count) {
    assert(mdata1.size() == mdata2.size());
    assert(mdata1.size() == cdata.rows());
    assert(mdata1.size() == cdata.cols());
    if (count <= 1) {
        return make_nans<double>(cdata.rows(), cdata.cols());
    }
    const auto cd = static_cast<double>(count);
    if constexpr (A == varalg::standard) {
        return (cdata - mdata1 * mdata2.transpose() / cd) / (cd - 1);
    } else {
        return cdata / (cd - 1);
    }
}

/** @} */

} // namespace simplemc::accs

#endif // SIMPLEMC_ACCS_UTILS_HPP
