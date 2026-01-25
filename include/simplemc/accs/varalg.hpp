/**
 * @file
 * @brief Tag that specifies how data is accumulated.
 */

#ifndef SIMPLEMC_ACCS_VARALG_HPP
#define SIMPLEMC_ACCS_VARALG_HPP

namespace simplemc {

/**
 * @addtogroup simplemc-accs-utils
 * @{
 */

/**
 * @brief Enumerate supported strategies how data can be accumulated.
 *
 * @details The following two strategies are available:
 * - `standard`: Simply sums up individual samples.
 * - `welford`: Uses the Welford algorithm or a variant thereof to accumulate the data (see
 * [Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)).
 *
 * See simplemc::accs::mean, simplemc::accs::diag_covariance and simplemc::accs::covariance for more
 * details how the choice of the algorithm affects the accumulation of the data.
 */
enum class varalg { standard, welford };

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VARALG_HPP
