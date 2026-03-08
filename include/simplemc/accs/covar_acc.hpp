/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a random vector.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_HPP

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/covar_acc_complex.hpp>
#include <simplemc/accs/covar_acc_double.hpp>
#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cstdint>
#include <optional>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs-covar
 * @{
 */

/**
 * @brief Accumulate (complex) random samples in a simplemc::covar_acc.
 *
 * @details See simplemc::covar_acc for more details on how the random samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::random_sample_range type.
 * @param rg Range containing the random samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::covar_acc containing the accumulated random samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_covar_acc(
    R&& rg, std::optional<ranges::range_value_t<R>> t = std::nullopt) { // NOLINT (ranges need not be forwarded)
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    if constexpr (double_or_complex<value_type>) {
        return detail::make_acc<covar_acc_single<value_type, A>>(rg, t, sz);
    } else {
        return detail::make_acc<covar_acc<value_type, A>>(rg, t, sz);
    }
}

/**
 * @brief Accumulate (complex) random samples in a simplemc::covar_acc wrapped in a
 * simplemc::block_acc.
 *
 * @details See simplemc::covar_acc and simplemc::block_acc for more details on how the random
 * samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::random_sample_range type.
 * @param rg Range containing the random samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param b Block size \f$ B \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::block_acc wrapping a simplemc::covar_acc containing the accumulated random
 * samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_block_covar_acc(R&& rg, std::uint64_t b, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    if constexpr (double_or_complex<value_type>) {
        return detail::make_acc<block_acc<covar_acc_single<value_type, A>>>(rg, t, b, sz);
    } else {
        return detail::make_acc<block_acc<covar_acc<value_type, A>>>(rg, t, b, sz);
    }
}

/**
 * @brief Accumulate (complex) random samples in a simplemc::covar_acc wrapped in a
 * simplemc::autocorr_acc.
 *
 * @details See simplemc::covar_acc and simplemc::autocorr_acc for more details on how the random
 * samples are accumulated.
 *
 * The autocorrelation accumulator uses the default multiplication factor \f$ c = 2 \f$ and minimum
 * number of levels \f$ L_{\text{min}} = 2 \f$.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::random_sample_range type.
 * @param rg Range containing the random samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::autocorr_acc wrapping a simplemc::covar_acc containing the accumulated random
 * samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_autocorr_covar_acc(R&& rg, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    if constexpr (double_or_complex<value_type>) {
        return detail::make_acc<autocorr_acc<covar_acc_single<value_type, A>>>(rg, t, sz);
    } else {
        return detail::make_acc<autocorr_acc<covar_acc<value_type, A>>>(rg, t, sz);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_HPP
