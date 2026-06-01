/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample variance of a data set.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_HPP
#define SIMPLEMC_ACCS_VAR_ACC_HPP

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc_complex.hpp>
#include <simplemc/accs/var_acc_double.hpp>
#include <simplemc/accs/var_acc_fwd.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cstdint>
#include <optional>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs-var
 * @{
 */

/**
 * @brief Accumulate (complex) data samples in a simplemc::var_acc.
 *
 * @details See @ref "simplemc::var_acc< X, A >" "simplemc::var_acc for real samples" and
 * @ref "simplemc::var_acc< Z, A >" "simplemc::var_acc for complex samples" for more details on how
 * the data samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::sample_range type.
 * @param rg Range containing the data samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::var_acc containing the accumulated data samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_var_acc(
    R&& rg, std::optional<ranges::range_value_t<R>> t = std::nullopt) { // NOLINT (ranges need not be forwarded)
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<var_acc<value_type, A>>(rg, t, sz);
}

/**
 * @brief Accumulate (complex) data samples in a simplemc::var_acc wrapped in a simplemc::block_acc.
 *
 * @details See @ref "simplemc::var_acc< X, A >" "simplemc::var_acc for real samples",
 * @ref "simplemc::var_acc< Z, A >" "simplemc::var_acc for complex samples" and simplemc::block_acc
 * for more details on how the data samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::sample_range type.
 * @param rg Range containing the data samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param b Block size \f$ B \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::block_acc wrapping a simplemc::var_acc containing the accumulated data samples
 * from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_block_var_acc(R&& rg, std::uint64_t b, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<block_acc<var_acc<value_type, A>>>(rg, t, b, sz);
}

/**
 * @brief Accumulate (complex) data samples in a simplemc::var_acc wrapped in a
 * simplemc::autocorr_acc.
 *
 * @details See @ref "simplemc::var_acc< X, A >" "simplemc::var_acc for real samples",
 * @ref "simplemc::var_acc< Z, A >" "simplemc::var_acc for complex samples" and simplemc::autocorr_acc
 * for more details on how the data samples are accumulated.
 *
 * The autocorrelation accumulator uses the default multiplication factor \f$ c = 2 \f$ and minimum
 * number of levels \f$ L_{\text{min}} = 2 \f$.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::sample_range type.
 * @param rg Range containing the data samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::autocorr_acc wrapping a simplemc::var_acc containing the accumulated data
 * samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_autocorr_var_acc(R&& rg, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<autocorr_acc<var_acc<value_type, A>>>(rg, t, sz);
}

/**
 * @brief Serialize a `var_acc` as `{"count": N, "mdata": [...], "cdata": [...]}`.
 *
 * @details Public getters + `(md, cd, n)` constructor; the template constraint allows both the real
 * and complex specializations since both expose the same public surface.
 */
template <class S, sample_type T, varalg A>
    requires serializer<std::remove_cvref_t<S>>
void simplemc_save(S&& s, const var_acc<T, A>& a) {
    s.save_at("count", a.count());
    s.save_at("mdata", a.mdata());
    s.save_at("cdata", a.cdata());
}

template <class S, sample_type T, varalg A>
    requires deserializer<std::remove_cvref_t<S>>
void simplemc_load(S&& s, var_acc<T, A>& a) {
    using va = var_acc<T, A>;
    typename va::count_type c {};
    auto md = a.mdata();
    auto cd = a.cdata();
    s.load_at("count", c);
    s.load_at("mdata", md);
    s.load_at("cdata", cd);
    a = va { md, cd, c };
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_HPP
