/**
 * @file
 * @brief Forward declaration of simplemc::covar_acc.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP

#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs
 * @{
 */

/**
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a random vector.
 *
 * @details The sample mean and the sample covariance matrix are used as approximations to the exact
 * expectation value and the covariance matrix, respectively. See @ref simplemc-accs for a definition
 * of those quantities.
 *
 * The covariance estimation does not account for any correlation between the samples. To get more
 * reliable results in case autocorrelation is a problem, see simplemc::block_acc,
 * simplemc::autocorr_acc or simplemc::batch_acc.
 *
 * The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * Please have a look at simplemc::accs::covariance.
 *
 * The implementation for real and complex random vectors is quite different. See the specializations
 * - @ref "simplemc::covar_acc< X, A >" "simplemc::covar_acc for real" and
 * - @ref "simplemc::covar_acc< Z, A >" "simplemc::covar_acc for cplx"
 * for more details.
 *
 * @tparam V simplemc::eigen_vector type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector V, varalg A = varalg::welford>
class covar_acc;

/**
 * @brief Alias for a statically sized covariance accumulator of size 1.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using covar_acc_single = covar_acc<Eigen::Matrix<T, 1, 1>, A>;

/**
 * @brief Alias for a statically sized covariance accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam M Size of the accumulator.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, int M, varalg A = varalg::welford>
    requires(M >= 1)
using covar_acc_static = covar_acc<Eigen::Matrix<T, M, 1>, A>;

/**
 * @brief Alias for a dynamically sized covariance accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using covar_acc_dynamic = covar_acc<Eigen::Matrix<T, Eigen::Dynamic, 1>, A>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP
