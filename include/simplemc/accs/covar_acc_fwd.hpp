/**
 * @file covar_acc_fwd.hpp
 * @brief Forward declaration of the covariance accumulator.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP

#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/concepts.hpp>

namespace simplemc {

/**
 * @brief Covariance accumulator for calculating arithmetic means and covariances.
 *
 * @details Naive estimation of the covariance matrix is available. It does not account for
 * any correlation between the samples.
 *
 * The user can choose between the standard and the more stable Welford algorithm and whether
 * to apply a constant shift to the accumulated data. This can sometimes improve the numerical
 * accuracy.
 *
 * If the size of the accumulator is 1, then values can be added with the stream operator:
 * @code{.cpp}
 * acc << val;
 * @endcode
 *
 * If the size of the accumulator is > 1 and only a single value should be added at once,
 * then one can use the subscript operator together with the stream operator:
 * @code{.cpp}
 * acc[idx] << val;
 * @endcode
 *
 * If the size of the accumulator is > 1 and multiple values should be added at once,
 * then one can use a multi value accumulator together with the stream operator:
 * @code{.cpp}
 * {
 *     auto mva = acc.make_mva();
 *     mva[idx1] << val1;
 *     mva[idx2] << val2;
 * }
 * @endcode
 *
 * If a range of values should be added at once, one can use the accumulate function:
 * @code{.cpp}
 * acc.accumulate(range, idx, count);
 * @endcode
 *
 * Results are always returned as Eigen::ArrayX or Eigen::ArrayXX objects. If e.g.
 * the size of the accumulator is 1, then we still need to access the array:
 * @code{.cpp}
 * auto mean = acc.mean()[0];
 * auto err = acc.stderror()[0];
 * @endcode
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to calculate the mean/variance.
 */
template <double_or_complex T, accs::varalg A = accs::varalg::standard>
class covar_acc;

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP
