/**
 * @file utils.hpp
 * @brief Utility functions for simplemc-accs.
 */

#ifndef SIMPLEMC_ACCS_UTILS_HPP
#define SIMPLEMC_ACCS_UTILS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <cstdint>

namespace simplemc::accs {

/**
 * @brief Enumerate the different algorithms how to calculate means, variances, etc.
 */
enum class varalg { standard, welford };

/**
 * @brief Make an Eigen::Array with NaNs.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @param size Size of storage.
 * @return Eigen::Array with NaNs.
 */
template <double_or_complex T>
Eigen::ArrayX<T> make_nans(long size);

/**
 * @brief Calculate the sample mean.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to calculate the mean.
 * @param data Accumulated values.
 * @param count Number of accumulated values.
 * @param shift Constant shift applied to accumulated values.
 * @return Sample mean.
 */
template <double_or_complex T, varalg A = varalg::standard>
Eigen::ArrayX<T> mean(const Eigen::ArrayX<T>& data, std::uint64_t count, T shift = 0.0);

} // namespace simplemc::accs

#endif // SIMPLEMC_ACCS_UTILS_HPP
