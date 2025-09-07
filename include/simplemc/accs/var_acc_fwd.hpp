/**
 * @file
 * @brief Forward declaration of simplemc::var_acc and type aliases.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_FWD_HPP
#define SIMPLEMC_ACCS_VAR_ACC_FWD_HPP

#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs-var
 * @{
 */

/**
 * @brief Class template for variance accumulators.
 *
 * @details Please see the specializations
 * - @ref "simplemc::var_acc< X, A >" "simplemc::var_acc for real" and
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
