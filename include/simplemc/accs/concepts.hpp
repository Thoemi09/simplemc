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
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief A concept that checks if a type can be used as a random sample for statistical analysis.
 *
 * @details `T` has to be either `double`, `std::complex<double>` or a simplemc::eigen_vector type.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept random_sample = double_or_complex<std::remove_cvref_t<T>> || eigen_vector<std::remove_cvref_t<T>>;

/**
 * @brief A concept that checks if a range contains simplemc::random_sample types.
 *
 * @details The range must be a sized range and a forward range.
 *
 * @tparam R Range type to check.
 */
template <typename R>
concept random_sample_range =
    ranges::sized_range<R> && ranges::forward_range<R> && random_sample<ranges::range_value_t<R>>;

/**
 * @brief Core accumulator interface for statistical data collection.
 *
 * @details An accumulator collects samples from the distribution of a (complex) random vector \f$
 * \mathbf{Z} \f$. A random sample \f$ \mathbf{z}^{(j)} \f$ is either \f$ \in \mathbb{R}^M \f$ or \f$
 * \in \mathbb{C}^M \f$, where \f$ M \geq 1 \f$ is the size of the random vector and \f$ j = 1, \dots,
 * N \f$ indexes the collected samples.
 *
 * Such an accumulator is fully defined by the type of random samples it accumulates.
 *
 * Let `a` be an instance of `A` and `s` be an instance of `A::sample_type`. The requirements for a
 * type `A` to be a basic accumulator are the following:
 *
 * - `A::value_type` is the scalar type of accumulated values, i.e. either \f$ \mathbb{R} \f$ or \f$
 * \mathbb{C} \f$ (simplemc::double_or_complex)
 * - `A::sample_type` is the type of the random sample/vector being accumulated, i.e. either \f$
 * \mathbb{R}^M \f$ or \f$ \mathbb{C}^M \f$ (simplemc::eigen_vector)
 * - `a.size()` returns \f$ M \f$, the number of elements of the random vector (integral type)
 * - `a.count()` returns \f$ N \f$, the number of samples accumulated so far (integral type)
 * - `a << s` accumulates a random sample `s` and returns a reference to `a`
 *
 * @tparam A Type to check.
 */
template <typename A>
concept basic_accumulator = requires {
    typename A::value_type;
    typename A::sample_type;
} && double_or_complex<typename A::value_type> && requires(A& acc, const A& cacc, typename A::sample_type s) {
    { cacc.size() } -> std::integral;
    { cacc.count() } -> std::integral;
    { acc << s } -> std::same_as<A&>;
};

/**
 * @brief Accumulator that computes sample means.
 *
 * @details Let `a` be an instance of `A`. The requirements for a type `A` to be a mean accumulator
 * are the following:
 *
 * - `A` satisfies simplemc::basic_accumulator
 * - `a.mean()` returns the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$ (see simplemc::accs::mean)
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
