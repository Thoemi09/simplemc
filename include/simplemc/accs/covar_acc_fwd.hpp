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
 * @brief Covariance accumulator for calculating the sample mean and covariance matrix of a 
 * random vector.
 *
 * @details Naive estimation of the covariance matrix is available. It does not account for
 * any correlation between the samples.
 *
 * The user can choose between the standard and the more stable Welford algorithm and whether
 * to apply a constant shift to the accumulated data. This can sometimes improve the numerical
 * accuracy.
 *
 * - If the size of the accumulator is 1, then values can be added with the stream operator:
 * @code{.cpp}
 * acc << val;
 * @endcode
 *
 * - If the size of the accumulator is > 1 and only a single value should be added at once,
 * then one can use the subscript operator together with the stream operator:
 * @code{.cpp}
 * acc[idx] << val;
 * @endcode
 *
 * - Multi value accumulators are not supported for the covariance accumulator (please use the
 * accumulate(..) function instead).
 *
 * - If a range of values should be added at once, one can use the accumulate function:
 * @code{.cpp}
 * acc.accumulate(range, idx);
 * @endcode
 * Here, `idx` is either a scalar denoting the starting index or a range of indices of the same
 * size as the range of values.
 *
 * Results are always returned as Eigen::VectorX or Eigen::MatrixX objects. If e.g. the size of the
 * accumulator is 1, then we still need to access the vector/matrix:
 * @code{.cpp}
 * auto mean = acc.mean()(0);
 * auto covariance = acc.covariance()(0,0);
 * @endcode
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm (either standard or welford).
 */
template <double_or_complex T, accs::varalg A = accs::varalg::standard>
class covar_acc;

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_FWD_HPP
