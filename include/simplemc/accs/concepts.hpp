/**
 * @file
 * @brief Concepts for **simplemc-accs**.
 */

#ifndef SIMPLEMC_ACCS_CONCEPTS_HPP
#define SIMPLEMC_ACCS_CONCEPTS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>

#include <concepts>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-concepts
 * @{
 */

/**
 * @brief A scalar data type supported by @ref simplemc-accs-accs.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept acc_scalar = double_or_complex<T>;

/**
 * @brief A vector data type supported by @ref simplemc-accs-accs.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept acc_vector = eigen_vector_dbl<T> || eigen_vector_cplx<T>;

/**
 * @brief A data type supported by @ref simplemc-accs-accs.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept acc_data = acc_scalar<T> || acc_vector<T>;

/**
 * @brief A range type containing simplemc::acc_data.
 *
 * @details In addition to its elements satisfying simplemc::acc_data, the range must also be a sized
 * and forward range.
 *
 * @tparam R Range type to check.
 */
template <typename R>
concept acc_data_range = ranges::sized_range<R> && ranges::forward_range<R> && acc_data<ranges::range_value_t<R>>;

/**
 * @brief A basic accumulator for statistical data collection.
 *
 * @details At the core of an accumulator is the ability to collect simplemc::acc_data.
 *
 * Let `acc` be an accumulator instance of type `A` and let `x` be an instance of `A::data_type`
 * satisfying simplemc::acc_data. The requirements for a type `A` to be a basic accumulator are the
 * following:
 *
 * - `A::data_type` is the type of accumulated data (simplemc::acc_data)
 * - `acc.count()` returns \f$ N \f$, the number of data points accumulated so far (integral type)
 * - `acc << x` accumulates a new data point `x` and returns a reference to `acc`
 *
 * @tparam A Type to check.
 */
template <typename A>
concept basic_accumulator = requires {
    typename A::data_type;
    requires acc_data<typename A::data_type>;
} && requires(A& acc, const A& cacc, typename A::data_type x) {
    { cacc.count() } -> std::integral;
    { acc << x } -> std::same_as<A&>;
};

/**
 * @brief Accumulator that computes sample means.
 *
 * @details Let `acc` be an instance of `A`. The requirements for a type `A` to be a mean accumulator
 * are the following:
 *
 * - `A` satisfies simplemc::basic_accumulator
 * - `acc.mean()` returns the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$
 *
 * See also @ref simplemc-accs-stats and simplemc::accs::mean for more background information.
 *
 * @tparam A Type to check.
 */
template <typename A>
concept mean_accumulator = basic_accumulator<A> && requires(const A& acc) {
    { acc.mean() };
};

/**
 * @brief Accumulator that computes sample means and variances.
 *
 * @details Let `a` be an instance of `A`. The requirements for a type `A` to be a variance
 * accumulator are the following:
 *
 * - `A` satisfies simplemc::mean_accumulator
 * - `a.variance()` returns the sample variance of the mean \f$ s_{\overline{\mathbf{X}}}^2 \f$ (see
 * simplemc::accs::diag_covariance)
 *
 * @tparam A Type to check.
 */
template <typename A>
concept variance_accumulator = mean_accumulator<A> && requires(const A& acc) {
    { acc.variance() };
};

/**
 * @brief Accumulator that computes sample means, variances, and covariances.
 *
 * @details Let `a` be an instance of `A`. The requirements for a type `A` to be a covariance
 * accumulator are the following:
 *
 * - `A` satisfies simplemc::variance_accumulator
 * - `a.covariance()` returns the sample covariance matrix of the mean \f$ s_{\overline{\mathbf{X}}
 * \overline{\mathbf{Y}}}^2 \f$ (see simplemc::accs::covariance)
 *
 * @tparam A Type to check.
 */
template <typename A>
concept covariance_accumulator = variance_accumulator<A> && requires(const A& acc) {
    { acc.covariance() };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_CONCEPTS_HPP
