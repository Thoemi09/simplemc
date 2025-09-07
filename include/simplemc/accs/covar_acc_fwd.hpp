/**
 * @file
 * @brief Forward declaration of simplemc::covar_acc and type aliases.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP

#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs-covar
 * @{
 */

/**
 * @brief Class template for covariance accumulators.
 *
 * @details Please see the specializations
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
