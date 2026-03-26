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
 * @brief Scalar data type supported by @ref simplemc-accs-accs.
 *
 * @details A type satisfies simplemc::sample_scalar if and only if it satisfies
 * simplemc::double_or_complex.
 *
 * Individual scalar samples are denoted as \f$ z^{(j)} \in \mathbb{C} \f$ or \f$ x^{(j)} \in
 * \mathbb{R} \f$. See also @ref simplemc-accs-stats.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept sample_scalar = double_or_complex<T>;

/**
 * @brief Vector data type supported by @ref simplemc-accs-accs.
 *
 * @details A type satisfies simplemc::sample_vector if and only if it satisfies
 * simplemc::eigen_vector_dbl or simplemc::eigen_vector_cplx.
 *
 * Individual vector samples are denoted as \f$ \mathbf{z}^{(j)} \in \mathbb{C}^M \f$ or \f$
 * \mathbf{x}^{(j)} \in \mathbb{R}^M \f$. See also @ref simplemc-accs-stats.
 *
 * Both fixed-size (e.g. `Eigen::Vector3d`) and dynamic-size (e.g. `Eigen::VectorXd`) vectors are
 * supported.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept sample_vector = eigen_vector_dbl<T> || eigen_vector_cplx<T>;

/**
 * @brief Data type supported by @ref simplemc-accs-accs.
 *
 * @details A type satisfies simplemc::sample_type if it is either a simplemc::sample_scalar or a
 * simplemc::sample_vector.
 *
 * All accumulators are parametrized by a type satisfying this concept. It determines whether the
 * accumulated data samples are scalars or vectors.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept sample_type = sample_scalar<T> || sample_vector<T>;

/**
 * @brief Range type containing simplemc::sample_type elements.
 *
 * @details A type satisfies simplemc::sample_range if it is a sized forward range whose elements
 * satisfy simplemc::sample_type.
 *
 * This corresponds to the data sets \f$ S_{\mathbf{Z}} \f$, \f$ S_{\mathbf{X}} \f$ pr \f$
 * S_{\mathbf{Y}} \f$ introduced in @ref simplemc-accs-stats.
 *
 * @tparam R Range type to check.
 */
template <typename R>
concept sample_range = ranges::sized_range<R> && ranges::forward_range<R> && sample_type<ranges::range_value_t<R>>;

/**
 * @brief Basic accumulator for statistical data collection.
 *
 * @details An accumulator collects data samples one at a time and maintains certain running
 * statistics depending on the type of the accumulator.
 *
 * Let `acc` be an accumulator instance of type `A` and let `x` be an instance of `A::sample_type`
 * satisfying simplemc::sample_type. The requirements for a type `A` to be a basic accumulator are the
 * following:
 *
 * - `A::sample_type` is the type of accumulated data (simplemc::sample_type)
 * - `A::value_type` is the underlying scalar type (simplemc::double_or_complex)
 * - `acc.count()` returns \f$ N \f$, the number of data points accumulated so far (integral type)
 * - `acc.size()` returns \f$ M \f$, the number of components of the accumulator (integral type)
 * - `acc << x` accumulates a new data point `x` and returns a reference to `acc`
 *
 * @tparam A Type to check.
 */
template <typename A>
concept basic_accumulator = requires {
    typename A::sample_type;
    requires sample_type<typename A::sample_type>;
    typename A::value_type;
    requires double_or_complex<typename A::value_type>;
} && requires(A& acc, const A& cacc, typename A::sample_type x) {
    { cacc.count() } -> std::integral;
    { cacc.size() } -> std::integral;
    { acc << x } -> std::same_as<A&>;
};

/**
 * @brief Accumulator that computes sample means.
 *
 * @details In addition to the requirements of simplemc::basic_accumulator, a mean accumulator
 * provides the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$ (see @ref simplemc-accs-stats-mean).
 *
 * Let `acc` be an instance of `A`. The requirements for a type `A` to be a mean accumulator are the
 * following:
 *
 * - `A` satisfies simplemc::basic_accumulator
 * - `acc.mean()` returns the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$
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
 * @details In addition to the requirements of simplemc::mean_accumulator, a variance accumulator
 * provides the sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$ (see
 * @ref simplemc-accs-stats-var and @ref simplemc-accs-stats-tau).
 *
 * Let `acc` be an instance of `A`. The requirements for a type `A` to be a variance accumulator are
 * the following:
 *
 * - `A` satisfies simplemc::mean_accumulator
 * - `acc.variance()` returns the sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$
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
 * @details In addition to the requirements of simplemc::variance_accumulator, a covariance
 * accumulator provides the full sample covariance matrix of the mean \f$ s_{\overline{\mathbf{Z}}
 * \overline{\mathbf{Z}}}^2 \f$ (see @ref simplemc-accs-stats-covar and @ref simplemc-accs-stats-tau).
 *
 * Let `acc` be an instance of `A`. The requirements for a type `A` to be a covariance accumulator are
 * the following:
 *
 * - `A` satisfies simplemc::variance_accumulator
 * - `acc.covariance()` returns the sample covariance matrix of the mean \f$ s_{\overline{\mathbf{Z}}
 * \overline{\mathbf{Z}}}^2 \f$
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
