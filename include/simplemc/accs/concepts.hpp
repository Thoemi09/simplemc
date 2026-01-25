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
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief Supported types that can be used as a scalar valued random variable (random scalar).
 *
 * @details A random variable \f$ Z \f$ is defined on a probability space \f$ (\Omega, \mathcal{F}, P)
 * \f$ and is a map \f$ Z: \Omega \to A \f$ from the sample space \f$ \Omega \f$ to some measurable
 * set \f$ A \f$.
 *
 * If the set \f$ A \f$ is either \f$ \mathbb{R} \f$ or \f$ \mathbb{C} \f$, then we call \f$ Z \f$ a
 * (complex) random scalar.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept random_scalar = double_or_complex<T>;

/**
 * @brief Supported types that can be used as a vector valued random variable (random vector).
 *
 * @details A random variable \f$ \mathbf{Z} \f$ is defined on a probability space \f$ (\Omega,
 * \mathcal{F}, P) \f$ and is a map \f$ \mathbf{Z}: \Omega \to A \f$ from the sample space \f$ \Omega
 * \f$ to some measurable set \f$ A \f$.
 *
 * If the set \f$ A \f$ is either \f$ \mathbb{R}^M \f$ or \f$ \mathbb{C}^M \f$, then we call \f$
 * \mathbf{Z} \f$ a (complex) random vector.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept random_vector = eigen_vector_dbl<T> || eigen_vector_cplx<T>;

/**
 * @brief Supported types that can be used as random samples for statistical analysis.
 *
 * @details A random sample \f$ z^{(j)} \f$ or \f$ \mathbf{z}^{(j)} \f$ is a sample taken from the
 * distribution of a (complex) simplemc::random_scalar \f$ Z \f$ or simplemc::random_vector \f$
 * \mathbf{Z} \f$, respectively.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept random_sample = random_scalar<T> || random_vector<T>;

/**
 * @brief Supported range types whose elements can be used as random samples for statistical analysis.
 *
 * @details A set of random samples \f$ S_Z = \left\{ z^{(j)} : j = 1, \dots, N \right\} \f$ or \f$ 
 * S_{\mathbf{Z}} = \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N \right\} \f$ contains \f$ N \f$
 * random samples taken from the distribution of a (complex) simplemc::random_scalar \f$ Z \f$ or
 * simplemc::random_vector \f$ \mathbf{Z} \f$, respectively.
 *
 * A range type that models such a set must be a sized, forward range and its elements must satisfy
 * simplemc::random_sample.
 *
 * @tparam R Range type to check.
 */
template <typename R>
concept random_sample_range =
    ranges::sized_range<R> && ranges::forward_range<R> && random_sample<ranges::range_value_t<R>>;

/**
 * @brief Basic accumulator interface for statistical data collection.
 *
 * @details An accumulator collects simplemc::random_sample objects taken from the distribution of a
 * (complex) simplemc::random_scalar \f$ Z \f$ or simplemc::random_vector \f$ \mathbf{Z} \f$.
 *
 * Let `acc` be an accumulator instance of type `A` and let `s` be an instance of `A::sample_type`. 
 * The requirements for a type `A` to be a basic accumulator are the following:
 *
 * - `A::value_type` is the scalar type of accumulated values (simplemc::double_or_complex)
 * - `A::sample_type` is the type of the random sample being accumulated (simplemc::random_sample)
 * - `acc.size()` returns \f$ M \f$, the number of elements of the random variable (integral type)
 * - `acc.count()` returns \f$ N \f$, the number of samples accumulated so far (integral type)
 * - `acc << s` accumulates a random sample `s` and returns a reference to `acc`
 *
 * @tparam A Type to check.
 */
template <typename A>
concept basic_accumulator = requires {
    typename A::value_type;
    typename A::sample_type;
    requires double_or_complex<typename A::value_type>;
    requires random_sample<typename A::sample_type>;
} && requires(A& acc, const A& cacc, typename A::sample_type s) {
    { cacc.size() } -> std::integral;
    { cacc.count() } -> std::integral;
    { acc << s } -> std::same_as<A&>;
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
 * See also @ref simplemc-accs-notes and simplemc::accs::mean for more background information.
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
