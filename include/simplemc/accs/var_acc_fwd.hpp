/**
 * @file
 * @brief Forward declaration of simplemc::var_acc.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_FWD_HPP
#define SIMPLEMC_ACCS_VAR_ACC_FWD_HPP

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
 * @brief Accumulator for calculating the sample mean and sample variance of a random vector.
 *
 * @details The sample mean and the sample variance are used as approximations to the exact
 * expectation value and diagonal of the covariance matrix, respectively. See @ref simplemc-accs for a
 * definition of those quantities.
 *
 * The variance estimation does not account for any correlation between the samples. To get more
 * reliable results in case autocorrelation is a problem, see simplemc::block_acc and
 * simplemc::autocorr_acc.
 *
 * The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * Please have a look at simplemc::accs::diag_covariance.
 *
 * The implementation for real and complex random vectors is quite different. See the specializations
 * - @ref "simplemc::var_acc< X, A >" "simplemc::var_acc for dbl" and
 * - @ref "simplemc::var_acc< Z, A >" "simplemc::var_acc for cplx"
 * for more details.
 *
 * @tparam V simplemc::eigen_vector type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector V, varalg A = varalg::welford>
class var_acc;

/**
 * @brief Alias for a statically sized variance accumulator of size 1.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using var_acc_single = var_acc<Eigen::Matrix<T, 1, 1>, A>;

/**
 * @brief Alias for a statically sized variance accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam M Size of the accumulator.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, int M, varalg A = varalg::welford>
    requires(M >= 1)
using var_acc_static = var_acc<Eigen::Matrix<T, M, 1>, A>;

/**
 * @brief Alias for a dynamically sized variance accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using var_acc_dynamic = var_acc<Eigen::Matrix<T, Eigen::Dynamic, 1>, A>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_FWD_HPP
