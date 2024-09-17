/**
 * @file
 * @brief Utilities for simplemc-accs.
 */

#ifndef SIMPLEMC_ACCS_UTILS_HPP
#define SIMPLEMC_ACCS_UTILS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>

#include <cassert>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-accs-utils
 * @brief Enumerate the different strategies how to accumulate data.
 *
 * @details The following two strategies are available:
 * - `standard`: Simply sums up individual samples.
 * - `welford`: Uses the Welford algorithm or a variant thereof to accumulate the data (see
 * [Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)).
 */
enum class varalg { standard, welford };

} // namespace simplemc

namespace simplemc::accs {

/**
 * @addtogroup simplemc-accs-utils
 * @{
 */

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
 * Here, \f$ \mathbf{t} \f$ is a constant vector that can optionally be applied to the random
 * samples to increase numerical accuracy.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector type.
 * @param mdata Accumulated mean data.
 * @param count Number of accumulated values.
 * @return Sample mean.
 */
template <varalg A, eigen_vector V>
[[nodiscard]] V mean(const V& mdata, std::uint64_t count) {
    if (count == 0) {
        return nans_vector<V>(mdata.size());
    }
    if constexpr (A == varalg::standard) {
        return mdata / static_cast<double>(count);
    } else {
        return mdata;
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
 * Here, \f$ \mathbf{t} \f$ and \f$ \mathbf{s} \f$ are constant vectors that can optionally be applied
 * to the random samples to increase numerical accuracy.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector_dbl type.
 * @param mdata1 Accumulated mean data of random vector \f$ \mathbf{X} \f$.
 * @param mdata2 Accumulated mean data of random vector \f$ \mathbf{Y} \f$.
 * @param cdata Accumulated (cross-)covariance data.
 * @param count Number of accumulated values.
 * @return Diagonal of the sample (cross-)covariance matrix.
 */
template <varalg A, eigen_vector_dbl V>
[[nodiscard]] V diag_covariance(
    [[maybe_unused]] const V& mdata1, [[maybe_unused]] const V& mdata2, const V& cdata, std::uint64_t count) {
    assert(mdata1.size() == mdata2.size());
    assert(mdata1.size() == cdata.size());
    if (count <= 1) {
        return nans_vector<V>(mdata1.size());
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
 * Here, \f$ \mathbf{t} \f$ and \f$ \mathbf{s} \f$ are constant vectors that can optionally be applied
 * to the random samples to increase numerical accuracy.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector_dbl type.
 * @tparam M simplemc::eigen_matrix_dbl type.
 * @param mdata1 Accumulated mean data of random vector \f$ \mathbf{X} \f$.
 * @param mdata2 Accumulated mean data of random vector \f$ \mathbf{Y} \f$.
 * @param cdata Accumulated (cross-)covariance data.
 * @param count Number of accumulated values.
 * @return Sample (cross-)covariance matrix.
 */
template <varalg A, eigen_vector_dbl V, eigen_matrix_dbl M>
[[nodiscard]] M covariance(
    [[maybe_unused]] const V& mdata1, [[maybe_unused]] const V& mdata2, const M& cdata, std::uint64_t count) {
    assert(mdata1.size() == mdata2.size());
    assert(mdata1.size() == cdata.rows());
    assert(mdata1.size() == cdata.cols());
    if (count <= 1) {
        return nans_matrix<M>(cdata.rows(), cdata.cols());
    }
    const auto cd = static_cast<double>(count);
    if constexpr (A == varalg::standard) {
        return (cdata - mdata1 * mdata2.transpose() / cd) / (cd - 1);
    } else {
        return cdata / (cd - 1);
    }
}

/**
 * @brief Calculate the integrated autocorrelation time for the elements of a (cross)-covariance
 * matrix.
 *
 * @details Inverting the last equation in @ref simplemc-accs, we can write the integrated
 * autocorrelation time as
 * \f[
 *   \tau_{\mathbf{X}\mathbf{Y}} = \frac{1}{2} \left( \frac{\mathrm{Cov}[\overline{\mathbf{X}}^{(N)},
 *   \overline{\mathbf{Y}}^{(N)}]}{\mathrm{Cov}[\mathbf{X}, \mathbf{Y}]} N - 1 \right) \; .
 * \f]
 * If we use a blocking method to determine the block size \f$ B \f$ at which the block averages,
 * \f$ \overline{\mathbf{X}}^{(B)} \f$ and \f$ \overline{\mathbf{Y}}^{(B)} \f$, become uncorrelated,
 * then this equation can be approximated with
 * \f[
 *   \tau_{\mathbf{X}\mathbf{Y}} \approx \frac{1}{2} \left( \frac{s_{\overline{\mathbf{X}}^{(B)}
 *   \overline{\mathbf{Y}}^{(B)}}^2}{s_{\mathbf{X}\mathbf{Y}}^2} B - 1 \right) \; .
 * \f]
 * where \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$ is the naive (unblocked) sample (cross)-covariance matrix
 * and \f$ s_{\overline{\mathbf{X}}^{(B)}\overline{\mathbf{Y}}^{(B)}}^2 \f$ is the sample
 * (cross)-covariance matrix of the blocked samples with block size \f$ B \f$.
 *
 * A similar equation in terms of variances holds for \f$ \tau_{X_i} \f$.
 *
 * @param c_naive Naive (unblocked) estimate of the sample (cross)-covariance matrix.
 * @param c_blocked Blocked estimate of the sample (cross)-covariance matrix.
 * @param blsize Block size used in the blocked estimate (w.r.t. the naive estimate).
 * @return Matrix of integrated autocorrelation times.
 */
template <eigen_matrix M>
[[nodiscard]] M tau(const M& c_naive, const M& c_blocked, std::uint64_t blsize) {
    assert(c_naive.rows() == c_blocked.rows());
    assert(c_naive.cols() == c_blocked.cols());
    return ((c_blocked.array() * blsize / c_naive.array() - 1.0) * 0.5).matrix();
}

/** @} */

} // namespace simplemc::accs

namespace simplemc::detail {

// Depending on the size and the memory location of an Eigen object, either a vector/matrix or a
// scalar is returned.
template <bool return_scalar, typename T>
auto scalar_or_matrix(T&& obj) {
    if constexpr (return_scalar) {
        return std::forward<T>(obj)(0, 0);
    } else {
        return std::forward<T>(obj);
    }
}

} // namespace simplemc::detail

#endif // SIMPLEMC_ACCS_UTILS_HPP
