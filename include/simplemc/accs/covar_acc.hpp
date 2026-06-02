/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a data set.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_HPP

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/covar_acc_complex.hpp>
#include <simplemc/accs/covar_acc_double.hpp>
#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/serialize/concepts.hpp>
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
 * @brief Accumulate (complex) data samples in a simplemc::covar_acc.
 *
 * @details See @ref "simplemc::covar_acc< X, A >" "simplemc::covar_acc for real samples" and
 * @ref "simplemc::covar_acc< Z, A >" "simplemc::covar_acc for complex samples" for more details on
 * how the data samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::sample_range type.
 * @param rg Range containing the data samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::covar_acc containing the accumulated data samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_covar_acc(
    R&& rg, std::optional<ranges::range_value_t<R>> t = std::nullopt) { // NOLINT (ranges need not be forwarded)
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<covar_acc<value_type, A>>(rg, t, sz);
}

/**
 * @brief Accumulate (complex) data samples in a simplemc::covar_acc wrapped in a simplemc::block_acc.
 *
 * @details See @ref "simplemc::covar_acc< X, A >" "simplemc::covar_acc for real samples",
 * @ref "simplemc::covar_acc< Z, A >" "simplemc::covar_acc for complex samples" and
 * simplemc::block_acc for more details on how the data samples are accumulated.
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
 * @return simplemc::block_acc wrapping a simplemc::covar_acc containing the accumulated data samples
 * from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_block_covar_acc(R&& rg, std::uint64_t b, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<block_acc<covar_acc<value_type, A>>>(rg, t, b, sz);
}

/**
 * @brief Accumulate (complex) data samples in a simplemc::covar_acc wrapped in a
 * simplemc::autocorr_acc.
 *
 * @details See @ref "simplemc::covar_acc< X, A >" "simplemc::covar_acc for real samples",
 * @ref "simplemc::covar_acc< Z, A >" "simplemc::covar_acc for complex samples" and
 * simplemc::autocorr_acc for more details on how the data samples are accumulated.
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
 * @return simplemc::autocorr_acc wrapping a simplemc::covar_acc containing the accumulated data
 * samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_autocorr_covar_acc(R&& rg, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<autocorr_acc<covar_acc<value_type, A>>>(rg, t, sz);
}

/**
 * @brief Serialize a covar_acc by its sample count, mean data, and covariance matrix.
 *
 * @details Uses public `count()` / `mdata()` / `cdata()` getters with the `(mdata, cdata, count)`
 * constructor. `cdata` is a matrix; the Eigen `adl_serializer` (in
 * `serialize/json/serializers.hpp`) handles its JSON round-trip on the backend side.
 *
 * @tparam S Serializer type.
 * @tparam T Sample type.
 * @tparam A Variance algorithm.
 * @param s Serializer.
 * @param acc Covariance accumulator to save.
 */
template <serializer S, sample_type T, varalg A>
void simplemc_save(S& s, const covar_acc<T, A>& acc) {
    s.save_at("count", acc.count());
    s.save_at("mdata", acc.mdata());
    s.save_at("cdata", acc.cdata());
}

/**
 * @brief Deserialize a covar_acc (inverse of @ref simplemc_save).
 *
 * @tparam S Deserializer type.
 * @tparam T Sample type.
 * @tparam A Variance algorithm.
 * @param s Deserializer.
 * @param acc Covariance accumulator to populate.
 */
template <deserializer S, sample_type T, varalg A>
void simplemc_load(const S& s, covar_acc<T, A>& acc) {
    using ca = covar_acc<T, A>;
    typename ca::count_type count {};
    auto mdata = acc.mdata();
    auto cdata = acc.cdata();
    s.load_at("count", count);
    s.load_at("mdata", mdata);
    s.load_at("cdata", cdata);
    acc = ca { mdata, cdata, count };
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_HPP
