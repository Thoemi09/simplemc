/**
 * @file
 * @brief Utilities for **simplemc-accs**.
 */

#ifndef SIMPLEMC_ACCS_UTILS_HPP
#define SIMPLEMC_ACCS_UTILS_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

namespace simplemc::accs {

/**
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief Calculate the sample mean of a (complex) random vector \f$ \mathbf{Z} \f$.
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
 * - `welford`: The mean data is accumulated with
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
 * @note The constant vector \f$ \mathbf{t} \f$ is not part of the function. It is the caller's
 * responsibility to subtract it while accumulating the data and to add it back after calculating the
 * mean.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector type.
 * @param mdata Accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$.
 * @param n Number of accumulated samples \f$ N \f$.
 * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
 */
template <varalg A, eigen_vector V>
[[nodiscard]] V mean(const V& mdata, std::uint64_t n) {
    if (n == 0) {
        return nans_vector<V>(mdata.size());
    }
    if constexpr (A == varalg::standard) {
        return mdata / static_cast<double>(n);
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
 *     \mathbf{c}^{(N)} = \mathbf{c}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{s} \right) \odot
 *     \left( \mathbf{y}^{(N)} - \mathbf{t} \right) =
 *     \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \mathbf{s} \right) \odot
 *     \left( \mathbf{y}^{(j)} - \mathbf{t} \right) \; ,
 *   \f]
 *   such that the diagonal of the sample (cross-)covariance matrix is given by
 *   \f[
 *     \mathrm{diag} \left( s_{\mathbf{X}\mathbf{Y}}^2 \right) = \frac{1}{N - 1} \left\{
 *     \mathbf{c}^{(N)} - \frac{\mathbf{m}_\mathbf{X}^{(N)} \odot \mathbf{m}_\mathbf{Y}^{(N)}}{N}
 *     \right\} = \frac{1}{N - 1} \left\{ \sum_{j=1}^N \mathbf{x}^{(j)} \odot \mathbf{y}^{(j)} -
 *     \frac{\sum_{j=1}^N \mathbf{x}^{(j)} \odot \sum_{k=1}^N \mathbf{y}^{(k)}}{N} \right\} \; .
 *   \f]
 * - `welford`: The mean data, \f$ \mathbf{n}_\mathbf{X}^{(N)} \f$ and
 * \f$ \mathbf{n}_\mathbf{Y}^{(N)} \f$, for the two real random vectors \f$ \mathbf{X} \f$ and
 * \f$ \mathbf{Y} \f$ is accumulated as stated in simplemc::accs::mean. The (cross-)covariance data is
 * accumulated with
 *   \f[
 *     \mathbf{d}^{(N)} = \mathbf{d}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{s} -
 *     \mathbf{n}_\mathbf{X}^{(N-1)} \right) \odot \left( \mathbf{y}^{(N)} - \mathbf{t} -
 *     \mathbf{n}_\mathbf{Y}^{(N)} \right)
 *     = \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \mathbf{s} - \mathbf{n}_\mathbf{X}^{(N)} \right)
 *     \odot \left( \mathbf{y}^{(j)} - \mathbf{t} - \mathbf{n}_\mathbf{Y}^{(N)} \right) \; ,
 *   \f]
 *   such that the diagonal of the sample (cross-)covariance matrix is given by
 *   \f[
 *     \mathrm{diag} \left( s_{\mathbf{X}\mathbf{Y}}^2 \right) = \frac{\mathbf{d}^{(N)}}{N - 1} =
 *     \frac{1}{N - 1} \left\{ \sum_{j=1}^N \mathbf{x}^{(j)} \odot \mathbf{y}^{(j)} -
 *     \frac{\sum_{j=1}^N \mathbf{x}^{(j)} \odot \sum_{k=1}^N \mathbf{y}^{(k)}}{N} \right\} \; .
 *   \f]
 *
 * Here, \f$ \mathbf{t} \f$ and \f$ \mathbf{s} \f$ are constant vectors that can optionally be applied
 * to the random samples to increase numerical accuracy and \f$ \mathbf{a} \odot \mathbf{b} \f$ is the
 * Hadamard (element-wise) product between two vectors \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$.
 *
 * @note The constant vectors \f$ \mathbf{t} \f$ and \f$ \mathbf{s} \f$ are not part of the function.
 * It is the caller's responsibility to subtract them while accumulating the data.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector_dbl type.
 * @param mdata_x Accumulated mean data \f$ \mathbf{m}_\mathbf{X}^{(N)}/\mathbf{n}_\mathbf{X}^{(N)}
 * \f$.
 * @param mdata_y Accumulated mean data \f$ \mathbf{m}_\mathbf{Y}^{(N)}/\mathbf{n}_\mathbf{Y}^{(N)}
 * \f$.
 * @param cdata Accumulated (cross-)covariance data \f$ \mathbf{c}^{(N)}/\mathbf{d}^{(N)} \f$.
 * @param n Number of accumulated samples \f$ N \f$.
 * @return Diagonal of the sample (cross-)covariance matrix \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$.
 */
template <varalg A, eigen_vector_dbl V>
[[nodiscard]] V diag_covariance(
    [[maybe_unused]] const V& mdata_x, [[maybe_unused]] const V& mdata_y, const V& cdata, std::uint64_t n) {
    assert(mdata_x.size() == mdata_y.size());
    assert(mdata_x.size() == cdata.size());
    if (n <= 1) {
        return nans_vector<V>(mdata_x.size());
    }
    const auto cd = static_cast<double>(n);
    if constexpr (A == varalg::standard) {
        return (cdata - mdata_x.cwiseProduct(mdata_y) / cd) / (cd - 1);
    } else {
        return cdata / (cd - 1);
    }
}

/**
 * @brief Calculate the full sample (cross-)covariance matrix of two real random vectors.
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
 *   such that the sample (cross-)covariance matrix is given by
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
 * @note The constant vectors \f$ \mathbf{t} \f$ and \f$ \mathbf{s} \f$ are not part of the function.
 * It is the caller's responsibility to subtract them while accumulating the data.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector_dbl type.
 * @tparam M simplemc::eigen_matrix_dbl type.
 * @param mdata_x Accumulated mean data \f$ \mathbf{m}_\mathbf{X}^{(N)}/\mathbf{n}_\mathbf{X}^{(N)}
 * \f$.
 * @param mdata_y Accumulated mean data \f$ \mathbf{m}_\mathbf{Y}^{(N)}/\mathbf{n}_\mathbf{Y}^{(N)}
 * \f$.
 * @param cdata Accumulated (cross-)covariance data \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$.
 * @param n Number of accumulated samples \f$ N \f$.
 * @return Sample (cross-)covariance matrix \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$.
 */
template <varalg A, eigen_vector_dbl V, eigen_matrix_dbl M>
[[nodiscard]] M covariance(
    [[maybe_unused]] const V& mdata_x, [[maybe_unused]] const V& mdata_y, const M& cdata, std::uint64_t n) {
    assert(mdata_x.size() == mdata_y.size());
    assert(mdata_x.size() == cdata.rows());
    assert(mdata_x.size() == cdata.cols());
    if (n <= 1) {
        return nans_matrix<M>(cdata.rows(), cdata.cols());
    }
    const auto cd = static_cast<double>(n);
    if constexpr (A == varalg::standard) {
        return (cdata - mdata_x * mdata_y.transpose() / cd) / (cd - 1);
    } else {
        return cdata / (cd - 1);
    }
}

/**
 * @brief Calculate the integrated autocorrelation time for the elements of a real (cross)-covariance
 * matrix or variance vector.
 *
 * @details Inverting the last equation in @ref simplemc-accs-stats, we can write the integrated
 * autocorrelation time as
 * \f[
 *   \tau_{\mathbf{X}\mathbf{Y}} = \frac{1}{2} \left( \frac{\mathrm{Cov}[\overline{\mathbf{X}}^{(N)},
 *   \overline{\mathbf{Y}}^{(N)}]}{\mathrm{Cov}[\mathbf{X}, \mathbf{Y}]} N - 1 \right) \approx
 *   \frac{1}{2} \left( \frac{s_{\overline{\mathbf{X}}^{(N)}\overline{\mathbf{Y}}^{(N)}}^2}
 *   {s_{\mathbf{X}\mathbf{Y}}^2} N - 1 \right) \; .
 * \f]
 * If we use a blocking method to determine the block size \f$ B \f$ at which the block averages,
 * \f$ \overline{\mathbf{X}}^{(B)} \f$ and \f$ \overline{\mathbf{Y}}^{(B)} \f$, become uncorrelated,
 * then this equation can be further approximated with
 * \f[
 *   \tau_{\mathbf{X}\mathbf{Y}} \approx \frac{1}{2} \left( \frac{s_{\overline{\mathbf{X}}^{(B)}
 *   \overline{\mathbf{Y}}^{(B)}}^2}{s_{\mathbf{X}\mathbf{Y}}^2} B - 1 \right) \; ,
 * \f]
 * where \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$ is the naive (unblocked) sample (cross)-covariance matrix
 * and \f$ s_{\overline{\mathbf{X}}^{(B)}\overline{\mathbf{Y}}^{(B)}}^2 =
 * s_{\overline{\mathbf{X}}^{(N)}\overline{\mathbf{Y}}^{(N)}}^2 N_{\mathrm{eff}} \f$ is the sample
 * (cross)-covariance matrix of the blocked samples with block size \f$ B = N / N_{\mathrm{eff}} \f$.
 *
 * A similar equation holds for \f$ \tau_{\mathbf{X}} \f$.
 *
 * @tparam M simplemc::eigen_matrix_dbl type.
 * @param s_naive Naive (unblocked) estimate of the sample (cross)-covariance matrix
 * \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$ or the sample variance \f$ s_{\mathbf{X}}^2 \f$.
 * @param s_blocked Blocked estimate of the sample (cross)-covariance matrix
 * \f$ s_{\overline{\mathbf{X}}^{(B)}\overline{\mathbf{Y}}^{(B)}}^2 \f$ or the sample variance
 * \f$ s_{\overline{\mathbf{X}}^{(B)}}^2 \f$.
 * @param b Block size \f$ B \f$ used in the blocked estimate (w.r.t. the naive estimate).
 * @return Matrix \f$ \tau_{\mathbf{X}\mathbf{Y}} \f$ or vector \f$ \tau_{\mathbf{X}} \f$ of
 * integrated autocorrelation times.
 */
template <eigen_matrix_dbl M>
[[nodiscard]] M tau(const M& s_naive, const M& s_blocked, std::uint64_t b) {
    assert(s_naive.rows() == s_blocked.rows());
    assert(s_naive.cols() == s_blocked.cols());
    return ((s_blocked.array() * b / s_naive.array() - 1.0) * 0.5).matrix();
}

/**
 * @brief Calculate the integrated autocorrelation time for a real scalar (cross)-covariance or
 * variance.
 *
 * @details See simplemc::accs::tau(const M&, const M&, std::uint64_t) for details.
 *
 * @param s_naive Naive (unblocked) estimate of the sample (cross)-covariance \f$ s_{XY}^2 \f$ or the
 * sample variance \f$ s_{X}^2 \f$.
 * @param s_blocked Blocked estimate of the sample (cross)-covariance \f$ s_{\overline{X}^{(B)}
 * \overline{Y}^{(B)}}^2 \f$ or the sample variance \f$ s_{\overline{X}^{(B)}}^2 \f$.
 * @param b Block size \f$ B \f$ used in the blocked estimate (w.r.t. the naive estimate).
 * @return Integrated autocorrelation time \f$ \tau_{XY} \f$ or \f$ \tau_{X} \f$.
 */
[[nodiscard]] inline double tau(double s_naive, double s_blocked, std::uint64_t b) {
    return (s_blocked * static_cast<double>(b) / s_naive - 1.0) * 0.5;
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

// Get the number of elements in a random sample.
template <sample_type T>
long random_sample_size([[maybe_unused]] const T& sample) {
    if constexpr (double_or_complex<T>) {
        return 1;
    } else {
        return sample.size();
    }
}

// Get a zero scalar/vector depending on the given sample type.
template <sample_type T>
auto zero_sample([[maybe_unused]] const T& sample) {
    if constexpr (double_or_complex<T>) {
        return T { 0.0 };
    } else {
        return T::Zero(sample.size());
    }
}

// Factory function to create an accumulator with the given random samples.
template <typename A, sample_range R, typename... Args>
[[nodiscard]] A make_acc(R&& rg, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t, Args&&... args) {
    using value_type = ranges::range_value_t<R>;
    using acc_type = A;

    if (ranges::size(rg) == 0) {
        throw simplemc_exception("Empty range of random samples");
    }

    acc_type acc { std::forward<Args>(args)... };
    const value_type t_shift = (t ? *t : zero_sample(*ranges::begin(rg)));
    for (const auto& sample : rg) {
        acc << (sample - t_shift);
    }
    return acc;
}

} // namespace simplemc::detail

#endif // SIMPLEMC_ACCS_UTILS_HPP
