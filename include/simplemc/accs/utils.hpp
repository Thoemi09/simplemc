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
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief Calculate the standard error of the mean from any variance accumulator.
 *
 * @details It computes \f$ s_{\overline{\mathbf{X}}} = \sqrt{s_{\overline{\mathbf{X}}}^2} \f$, where
 * \f$ s_{\overline{\mathbf{X}}}^2 \f$ is the variance returned by the accumulator's `variance()`
 * method. For vector-valued accumulators, the square root is applied component-wise.
 *
 * @tparam A Type satisfying simplemc::variance_accumulator.
 * @param acc Variance accumulator.
 * @return Standard error of the mean \f$ s_{\overline{\mathbf{X}}} \f$.
 */
template <variance_accumulator A>
[[nodiscard]] auto stderror(const A& acc) {
    if constexpr (sample_scalar<typename A::sample_type>) {
        return std::sqrt(acc.variance());
    } else {
        return acc.variance().cwiseSqrt().eval();
    }
}

/** @} */

} // namespace simplemc

namespace simplemc::accs {

/**
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief Calculate the sample mean of some (complex) data.
 *
 * @details The function expects that the data has already been accumulated according to the specified
 * simplemc::varalg.
 *
 * See @ref simplemc-accs-stats-mean for some background information and simplemc::mean_acc for
 * algorithmic details.
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
 * @brief Calculate the diagonal elements of the sample (cross-)covariance matrix of some real data.
 *
 * @details The function expects that the data has already been accumulated according to the specified
 * simplemc::varalg.
 *
 * Let \f$ S_{\mathbf{X}} \f$ and \f$ S_{\mathbf{Y}} \f$ be two data sets. This function calculates
 * the diagonal of the (cross)-covariance matrix, i.e. \f$ \mathrm{diag}(s_{\mathbf{X}\mathbf{Y}}^2)
 * \f$. In case, the two data sets are the same, this corresponds to the sample variance
 * \f$ s_{\mathbf{X}}^2 \f$ of the data set.
 *
 * See @ref simplemc-accs-stats-var for some background information and
 * @ref "simplemc::var_acc< X, A >" for algorithmic details.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector_dbl type.
 * @param mdata_x Accumulated mean data \f$ \mathbf{m}_\mathbf{X}^{(N)}/\mathbf{n}_\mathbf{X}^{(N)}
 * \f$.
 * @param mdata_y Accumulated mean data \f$ \mathbf{m}_\mathbf{Y}^{(N)}/\mathbf{n}_\mathbf{Y}^{(N)}
 * \f$.
 * @param cdata Accumulated (cross-)covariance data \f$ \mathbf{c}^{(N)}/\mathbf{d}^{(N)} \f$.
 * @param n Number of accumulated samples \f$ N \f$.
 * @return Diagonal of the sample (cross-)covariance matrix, i.e.\f$ \mathrm{diag}(s_{\mathbf{X}
 * \mathbf{Y}}^2) \f$.
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
 * @details The function expects that the data has already been accumulated according to the specified
 * simplemc::varalg.
 *
 * Let \f$ S_{\mathbf{X}} \f$ and \f$ S_{\mathbf{Y}} \f$ be two data sets. This function calculates
 * the full (cross-)covariance matrix \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$. In case, the two data sets
 * are the same, this corresponds to the sample covariance matrix \f$ s_{\mathbf{X}\mathbf{X}}^2 \f$
 * of the data set.
 *
 * See @ref simplemc-accs-stats-covar for some background information and
 * @ref "simplemc::covar_acc< X, A >" for algorithmic details.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam V simplemc::eigen_vector_dbl type.
 * @tparam M simplemc::eigen_matrix_dbl type.
 * @param mdata_x Accumulated mean data
 * \f$ \mathbf{m}_\mathbf{X}^{(N)}/\mathbf{n}_\mathbf{X}^{(N)} \f$.
 * @param mdata_y Accumulated mean data
 * \f$ \mathbf{m}_\mathbf{Y}^{(N)}/\mathbf{n}_\mathbf{Y}^{(N)} \f$.
 * @param cdata Accumulated (cross-)covariance data
 * \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$.
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
 * @brief Calculate the integrated autocorrelation time for the elements of a real (cross-)covariance
 * matrix or variance vector.
 *
 * @details The integrated autocorrelation time quantifies the degree of correlation between
 * successive samples. It is estimated by comparing a naive (unblocked) and a blocked estimate of the
 * sample (cross-)covariance matrix or variance.
 *
 * See @ref simplemc-accs-stats-tau for some background information and simplemc::autocorr_acc for
 * algorithmic details.
 *
 * @tparam M simplemc::eigen_matrix_dbl type.
 * @param s_naive Naive (unblocked) estimate of the sample (cross-)covariance matrix \f$ s_{\mathbf{X}
 * \mathbf{Y}}^2 \f$ or the sample variance \f$ s_{\mathbf{X}}^2 \f$.
 * @param s_blocked Blocked estimate of the sample (cross-)covariance matrix \f$ s_{\overline{
 * \mathbf{X}}^{(B)}\overline{\mathbf{Y}}^{(B)}}^2 \f$ or the sample variance \f$ s_{\overline{
 * \mathbf{X}}^{(B)}}^2 \f$.
 * @param b Block size \f$ B \f$ used in the blocked estimate (w.r.t. the naive estimate).
 * @return Matrix \f$ \tau_{\mathbf{X}\mathbf{Y}} \f$ or vector \f$ \tau_{\mathbf{X}} \f$ of
 * integrated autocorrelation times.
 */
template <eigen_matrix_dbl M>
[[nodiscard]] M tau(const M& s_naive, const M& s_blocked, std::uint64_t b) noexcept {
    assert(s_naive.rows() == s_blocked.rows());
    assert(s_naive.cols() == s_blocked.cols());
    return ((s_blocked.array() * b / s_naive.array() - 1.0) * 0.5).matrix();
}

/**
 * @brief Calculate the integrated autocorrelation time for a real scalar (cross-)covariance or
 * variance.
 *
 * @details Scalar overload of simplemc::accs::tau(const M&, const M&, std::uint64_t).
 *
 * @param s_naive Naive (unblocked) estimate of the sample (cross)-covariance \f$ s_{XY}^2 \f$ or the
 * sample variance \f$ s_{X}^2 \f$.
 * @param s_blocked Blocked estimate of the sample (cross)-covariance \f$ s_{\overline{X}^{(B)}
 * \overline{Y}^{(B)}}^2 \f$ or the sample variance \f$ s_{\overline{X}^{(B)}}^2 \f$.
 * @param b Block size \f$ B \f$ used in the blocked estimate (w.r.t. the naive estimate).
 * @return Integrated autocorrelation time \f$ \tau_{XY} \f$ or \f$ \tau_{X} \f$.
 */
[[nodiscard]] inline double tau(double s_naive, double s_blocked, std::uint64_t b) noexcept {
    return (s_blocked * static_cast<double>(b) / s_naive - 1.0) * 0.5;
}

/** @} */

} // namespace simplemc::accs

namespace simplemc::detail {

// Depending on the size and the memory location of an Eigen object, either a vector/matrix or a
// scalar is returned.
template <bool return_scalar, typename T>
auto scalar_or_matrix(T&& obj) noexcept {
    if constexpr (return_scalar) {
        return std::forward<T>(obj)(0, 0);
    } else {
        return std::forward<T>(obj);
    }
}

// Get the number of elements in a random sample.
template <sample_type T>
long random_sample_size([[maybe_unused]] const T& sample) noexcept {
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
