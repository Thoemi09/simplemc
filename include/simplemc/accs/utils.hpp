/**
 * @file
 * @brief Utilities functions for simplemc-accs.
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
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief Enumerate the different strategies how to accumulate data.
 *
 * @details The following two strategies are available:
 * - `standard`: Simply sums up individual samples.
 * - `welford`: Uses the Welford algorithm or a variant thereof to accumulate the data (see
 * [Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)).
 */
enum class varalg { standard, welford };

/**
 * @brief Make an `Eigen::VectorX` with NaNs.
 *
 * @tparam T Value type (either `double` or `std::complex<double>`).
 * @param size Size of the vector.
 * @return `Eigen::VectorX` with NaNs.
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
 * @brief Make an `Eigen::MatrixX` with NaNs.
 *
 * @tparam T Value type (either `double` or `std::complex<double>`).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return `Eigen::MatrixX` with NaNs.
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
 * @brief Calculate the sample mean of a (complex) random vector.
 *
 * @details The calculation of the sample mean depends on the algorithm used to accumulate the data:
 * - `standard`: The mean data is accumulated with
 *   \f[
 *     \mathbf{m}^{(N)} = \mathbf{m}^{(N-1)} + \mathbf{z}^{(N)} - \mathbf{t} =
 *     \sum_{j=1}^N \left( \mathbf{z}^{(j)} - \mathbf{t} \right) \; ,
 *   \f]
 *   such that the sample mean is given by
 *   \f[
 *     \overline{\mathbf{z}}^{(N)} = \frac{\mathbf{m}^{(N)}}{N} + \mathbf{t} =
 *     \frac{1}{N} \sum_{j=1}^N \mathbf{z}^{(j)} \; .
 *   \f]
 *
 * - `welford`: The mean data is accumlated with
 *   \f[
 *     \mathbf{n}^{(N)} = \mathbf{n}^{(N-1)} + \frac{1}{N} \left( \mathbf{z}^{(N)} - \mathbf{t} -
 *     \mathbf{n}^{(N-1)} \right) =
 *     \frac{1}{N} \sum_{j=1}^N \left( \mathbf{z}^{(j)} - \mathbf{t} \right) \; ,
 *   \f]
 *   such that the sample mean is given by
 *   \f[
 *     \overline{\mathbf{z}}^{(N)} = \mathbf{n}^{(N)} + \mathbf{t} =
 *     \frac{1}{N} \sum_{j=1}^N \mathbf{z}^{(j)} \; .
 *   \f]
 *
 * @tparam T Type of accumulated values (either `double` or `std::complex<double>`).
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
 * @brief Calculate the diagonal elements of the sample (cross-)covariance matrix of two real random
 * vectors.
 *
 * @details The calculation of the diagonal elements of the sample (cross-)covariance matrix depends
 * on the algorithm used to accumulate the data:
 * - `standard`: The mean data, \f$ \mathbf{m}_\mathbf{X}^{(N)} \f$ and
 * \f$ \mathbf{m}_\mathbf{Y}^{(N)} \f$, for the two real random vectors \f$ \mathbf{X} \f$ and
 * \f$ \mathbf{Y} \f$ is accumulated as stated in simplemc::accs::mean. The (cross-)covariance data is
 * accumulated with
 *   \f[
 *     \left( \mathbf{C}^{(N)} \right)_{ii} = c_i^{(N)} =
 *     c_i^{(N-1)} + \left( x_i^{(N)} - s_i \right) \left( y_i^{(N)} - t_i \right) =
 *     \sum_{j=1}^N \left( x_i^{(j)} - s_i \right) \left( y_i^{(j)} - t_i \right) \; ,
 *   \f]
 *   such that the diagonal of the sample (cross-)covariance matrix is given by
 *   \f[
 *     \left( s_{\mathbf{X}\mathbf{Y}}^2 \right)_{ii} = \frac{1}{N - 1} \left\{ c_i^{(N)} -
 *     \frac{\left( \mathbf{m}_\mathbf{X}^{(N)} \right)_i
 *     \left( \mathbf{m}_\mathbf{Y}^{(N)} \right)_i}{N} \right\} =
 *     \frac{1}{N - 1} \left\{ \sum_{j=1}^N x_i^{(j)} y_i^{(j)} - \frac{\sum_{j=1}^N x_i^{(j)}
 *     \sum_{k=1}^N y_i^{(k)}}{N} \right\} \; .
 *   \f]
 * - `welford`: The mean data, \f$ \mathbf{n}_\mathbf{X}^{(N)} \f$ and
 * \f$ \mathbf{n}_\mathbf{Y}^{(N)} \f$, for the two real random vectors \f$ \mathbf{X} \f$ and
 * \f$ \mathbf{Y} \f$ is accumulated as stated in simplemc::accs::mean. The (cross-)covariance data is
 * accumulated with
 *   \f{eqnarray*}{
 *     \left( \mathbf{D}^{(N)} \right)_{ii} &=& d_i^{(N)} =
 *     d_i^{(N-1)} + \left( x_i^{(N)} - s_i - \left( \mathbf{n}_\mathbf{X}^{(N-1)} \right)_i \right)
 *     \left( y_i^{(N)} - t_i - \left( \mathbf{n}_\mathbf{Y}^{(N)} \right)_i \right) \\
 *     &=& \sum_{j=1}^N \left( x_i^{(j)} - s_i - \left( \mathbf{n}_\mathbf{X}^{(N)} \right)_i \right)
 *     \left( y_i^{(j)} - t_i - \left( \mathbf{n}_\mathbf{Y}^{(N)} \right)_i \right) \; ,
 *   \f}
 *   such that the diagonal of the sample (cross-)covariance matrix is given by
 *   \f[
 *     \left( s_{\mathbf{X}\mathbf{Y}}^2 \right)_{ii} = \frac{d_i^{(N)}}{N - 1} =
 *     \frac{1}{N - 1} \left\{ \sum_{j=1}^N x_i^{(j)} y_i^{(j)} - \frac{\sum_{j=1}^N x_i^{(j)}
 *     \sum_{k=1}^N y_i^{(k)}}{N} \right\} \; .
 *   \f]
 *
 * @tparam A Algorithm used to accumulate the data.
 * @param mdata1 Accumulated mean data of random vector \f$ \mathbf{X} \f$.
 * @param mdata2 Accumulated mean data of random vector \f$ \mathbf{Y} \f$.
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
 * @details The calculation of the sample (cross-)covariance matrix depends on the algorithm used to
 * accumulate the data:
 * - `standard`: The mean data, \f$ \mathbf{m}_\mathbf{X}^{(N)} \f$ and
 * \f$ \mathbf{m}_\mathbf{Y}^{(N)} \f$, for the two real random vectors \f$ \mathbf{X} \f$ and
 * \f$ \mathbf{Y} \f$ is accumulated as stated in simplemc::accs::mean. The (cross-)covariance data is
 * accumulated with
 *   \f[
 *     \mathbf{C}^{(N)} =
 *     \mathbf{C}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{s} \right) \left( \mathbf{y}^{(N)} -
 *     \mathbf{t} \right)^T =
 *     \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \mathbf{s} \right) \left( \mathbf{y}^{(j)} - \mathbf{t}
 *     \right)^T \; ,
 *   \f]
 *   such that the sample (cross-)covariance matrix is given by
 *   \f[
 *     s_{\mathbf{X}\mathbf{Y}}^2 = \frac{1}{N - 1} \left\{ \mathbf{C}^{(N)} -
 *     \frac{\mathbf{m}_\mathbf{X}^{(N)} \left( \mathbf{m}_\mathbf{Y}^{(N)} \right)^T}{N} \right\} =
 *     \frac{1}{N - 1} \left\{ \sum_{j=1}^N \mathbf{x}^{(j)} \left( \mathbf{y}^{(j)} \right)^T -
 *     \frac{\sum_{j=1}^N \mathbf{x}^{(j)} \left( \sum_{k=1}^N \mathbf{y}^{(k)} \right)^T}{N} \right\}
 *     \; .
 *   \f]
 * - `welford`: The mean data, \f$ \mathbf{n}_\mathbf{X}^{(N)} \f$ and
 * \f$ \mathbf{n}_\mathbf{Y}^{(N)} \f$, for the two real random vectors \f$ \mathbf{X} \f$ and
 * \f$ \mathbf{Y} \f$ is accumulated as stated in simplemc::accs::mean. The (cross-)covariance data is
 * accumulated with
 *   \f{eqnarray*}{
 *     \mathbf{D}^{(N)} &=&
 *     \mathbf{D}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{s} - \mathbf{n}_\mathbf{X}^{(N-1)}
 *     \right) \left( \mathbf{y}^{(N)} - \mathbf{t} - \mathbf{n}_\mathbf{Y}^{(N)} \right)^T \\
 *     &=& \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \mathbf{s} - \mathbf{n}_\mathbf{X}^{(N)} \right)
 *     \left( \mathbf{y}^{(j)} - \mathbf{t} - \mathbf{n}_\mathbf{Y}^{(N)} \right)^T \; ,
 *   \f}
 *   such that the diagonal of the sample (cross-)covariance matrix is given by
 *   \f[
 *     s_{\mathbf{X}\mathbf{Y}}^2 = \frac{\mathbf{D}^{(N)}}{N - 1} =
 *     \frac{1}{N - 1} \left\{ \sum_{j=1}^N \mathbf{x}^{(j)} \left( \mathbf{y}^{(j)} \right)^T -
 *     \frac{\sum_{j=1}^N \mathbf{x}^{(j)} \left( \sum_{k=1}^N \mathbf{y}^{(k)} \right)^T}{N}
 *     \right\} \; .
 *   \f]
 *
 * @tparam A Algorithm used to accumulate the data.
 * @param mdata1 Accumulated mean data of random vector \f$ \mathbf{X} \f$.
 * @param mdata2 Accumulated mean data of random vector \f$ \mathbf{Y} \f$.
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
