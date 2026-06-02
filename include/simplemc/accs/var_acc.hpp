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
 * @brief Serialize a simplemc::var_acc.
 *
 * @details It serializes the number of accumulated samples \f$ N \f$ together with the accumulated
 * mean data and variance data.
 *
 * @tparam S simplemc::serializer type.
 * @tparam T simplemc::sample_type of the variance accumulator.
 * @tparam A simplemc::varalg algorithm of the variance accumulator.
 * @param s Serializer object.
 * @param acc Variance accumulator to serialize.
 */
template <serializer S, sample_type T, varalg A>
void simplemc_save(S& s, const var_acc<T, A>& acc) {
    s.save_at("count", acc.count());
    s.save_at("mdata", acc.mdata());
    s.save_at("cdata", acc.cdata());
    if constexpr (std::same_as<T, std::complex<double>> || eigen_vector_cplx<T>) {
        s.save_at("rdata", acc.rdata());
        s.save_at("idata", acc.idata());
    }
}

/**
 * @brief Deserialize a simplemc::var_acc.
 *
 * @details It first deserializes the number of accumulated samples \f$ N \f$ together with the
 * accumulated mean data and variance data and then uses them to construct the variance accumulator
 * (see simplemc::var_acc(const dbl_vec_type&, const dbl_vec_type&, count_type) for real samples and
 * simplemc::var_acc(const cplx_vec_type&, const dbl_vec_type&, const dbl_vec_type&,
 * const dbl_vec_type&, count_type) for complex samples).
 *
 * @tparam S simplemc::deserializer type.
 * @tparam T simplemc::sample_type of the variance accumulator.
 * @tparam A simplemc::varalg algorithm of the variance accumulator.
 * @param s Deserializer object.
 * @param acc Variance accumulator to deserialize into.
 */
template <deserializer S, sample_type T, varalg A>
void simplemc_load(const S& s, var_acc<T, A>& acc) {
    using acc_type = var_acc<T, A>;
    auto count = typename acc_type::count_type {};
    auto mdata = acc.mdata();
    auto cdata = acc.cdata();
    s.load_at("count", count);
    s.load_at("mdata", mdata);
    s.load_at("cdata", cdata);
    if constexpr (std::same_as<T, std::complex<double>> || eigen_vector_cplx<T>) {
        auto rdata = acc.rdata();
        auto idata = acc.idata();
        s.load_at("rdata", rdata);
        s.load_at("idata", idata);
        acc = acc_type { mdata, rdata, idata, cdata, count };
    } else {
        acc = acc_type { mdata, cdata, count };
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_HPP
