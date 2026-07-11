// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Jackknife resampling for accumulators.
 */

#ifndef SIMPLEMC_ACCS_JACKKNIFE_HPP
#define SIMPLEMC_ACCS_JACKKNIFE_HPP

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-resampling
 * @{
 */

/**
 * @brief Result of a jackknife resampling procedure.
 *
 * @details It stores the naive estimate \f$ f_{\text{naive}} = f(\overline{\mathbf{z}_1}^{(N)},
 * \overline{\mathbf{z}_2}^{(N)}, \dots) \f$ and a simplemc::covar_acc of \f$ N \f$ jackknife
 * samples \f$ f_{[i]}^{(N-1)} \f$.
 *
 * Furthermore, it provides convenience methods that apply the correct jackknife scaling factors to
 * obtain estimates for
 * - the standard error \f$ s_f \f$,
 * - the variance \f$ s_f^2 \f$,
 * - the covariance matrix \f$ s_{\mathbf{f} \mathbf{f}}^2 \f$,
 * - the jackknife bias estimate \f$ b_{\text{JK}} \f$, and
 * - the bias-corrected estimate \f$ \overline{f}_{\text{corr}} \f$.
 *
 * See @ref simplemc-accs-resampling for more details.
 *
 * @note In practice, this class is only constructed by simplemc::jackknife(), which always
 * wraps a simplemc::covar_acc. The template parameter exists so that downstream code can
 * spell the result type, not to support alternative accumulators.
 *
 * @tparam A Accumulator type satisfying simplemc::covariance_accumulator.
 */
template <covariance_accumulator A>
class jackknife_result {
public:
    /**
     * @brief Type of the stored accumulator.
     */
    using acc_type = A;

    /**
     * @brief Type of accumulated samples (simplemc::sample_type).
     */
    using sample_type = typename acc_type::sample_type;

    /**
     * @brief Underlying scalar type of accumulated samples (simplemc::double_or_complex).
     */
    using value_type = typename acc_type::value_type;

    /**
     * @brief Type for counting the number of accumulated values.
     */
    using count_type = typename acc_type::count_type;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = typename acc_type::size_type;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = acc_type::static_size;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = acc_type::is_dynamic;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return simplemc::varalg tag of the stored accumulator.
     */
    static constexpr auto varalg() noexcept { return A::varalg(); }

public:
    /**
     * @brief Construct a jackknife result.
     *
     * @param f_naive Naive estimate \f$ f_{\text{naive}} \f$.
     * @param acc Covariance accumulator of jackknife samples \f$ f_{[i]}^{(N-1)} \f$.
     */
    jackknife_result(sample_type f_naive, A acc) : f_naive_(std::move(f_naive)), acc_(std::move(acc)) {}

    /**
     * @brief Get the stored accumulator.
     *
     * @return Const reference to the stored covariance accumulator.
     */
    [[nodiscard]] const acc_type& accumulator() const noexcept { return acc_; }

    /**
     * @brief Get the number \f$ N \f$ of accumulated samples.
     *
     * @return Number of jackknife estimates \f$ f_{[i]}^{(N-1)} \f$.
     */
    [[nodiscard]] auto count() const noexcept { return acc_.count(); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulated samples.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const noexcept { return acc_.size(); }

    /**
     * @brief Get the naive estimate \f$ f_{\text{naive}} \f$ from the full data set.
     *
     * @details This is the value obtained by applying the transformation function to the means of the
     * original data, i.e. \f$ f_{\text{naive}} = f(\overline{\mathbf{z}_1}^{(N)},
     * \overline{\mathbf{z}_2}^{(N)}, \dots) \f$.
     *
     * @return Naive estimate \f$ f_{\text{naive}} \f$.
     */
    [[nodiscard]] const sample_type& naive_mean() const noexcept { return f_naive_; }

    /**
     * @brief Calculate the jackknife mean \f$ \overline{f}_{\text{JK}} \f$.
     *
     * @details It computes \f$ \overline{f}_{\text{JK}} = \frac{1}{N} \sum_{i=1}^N f_{[i]}^{(N-1)}
     * \f$.
     *
     * @return Jackknife mean \f$ \overline{f}_{\text{JK}} \f$.
     */
    [[nodiscard]] auto mean() const { return acc_.mean(); }

    /**
     * @brief Calculate the jackknife variance estimate \f$ s_f^2 \f$.
     *
     * @details It computes \f$ s_f^2 = (N-1)^{2} \, s_{\overline{f}}^{2} \f$, where \f$
     * s_{\overline{f}}^{2} \f$ is the sample variance of the mean of the jackknife samples \f$
     * f_{[i]}^{(N-1)} \f$. Equivalently, \f$ s_f^2 = \frac{N-1}{N} \sum_{i=1}^N \left( f_{[i]}^{
     * (N-1)} - \overline{f}_{\text{JK}} \right)^2 \f$ (see @ref simplemc-accs-resampling).
     *
     * For complex samples, the result is the real-valued error-bar quantity \f$ \mathrm{Var}
     * (\mathrm{Re}\,f) + \mathrm{Var}(\mathrm{Im}\,f) \f$. For vector complex samples this means
     * taking the real part of the (provably real) diagonal of the complex covariance matrix.
     *
     * @return Jackknife variance estimate \f$ s_f^2 \f$.
     */
    [[nodiscard]] auto variance() const {
        const auto nm1 = static_cast<double>(acc_.count() - 1);
        const auto scale = nm1 * nm1;
        if constexpr (sample_scalar<sample_type>) {
            return scale * acc_.variance();
        } else {
            return (scale * acc_.variance()).real().eval();
        }
    }

    /**
     * @brief Calculate the jackknife standard error estimate \f$ s_f \f$.
     *
     * @details It computes \f$ s_f = \sqrt{s_f^2} \f$, where \f$ s_f^2 \f$ is the jackknife variance
     * estimate returned by variance(). For vector-valued samples, the square root is applied
     * component-wise.
     *
     * @return Jackknife standard error estimate \f$ s_f \f$.
     */
    [[nodiscard]] auto stderror() const {
        if constexpr (sample_scalar<sample_type>) {
            return std::sqrt(variance());
        } else {
            return variance().cwiseSqrt().eval();
        }
    }

    /**
     * @brief Calculate the jackknife covariance estimate \f$ s_{\mathbf{f}\mathbf{f}}^2 \f$.
     *
     * @details It computes \f$ s_{\mathbf{f}\mathbf{f}}^2 = (N-1)^{2} \, s_{\overline{\mathbf{f}}
     * \overline{\mathbf{f}}}^{2} \f$, where \f$ s_{\overline{\mathbf{f}}\overline{\mathbf{f}}}^{2}
     * \f$ is the sample covariance matrix of the mean of the jackknife samples \f$ f_{[i]}^{(N-1)}
     * \f$.
     *
     * Unlike variance(), the full complex covariance matrix is returned for vector complex samples
     * (see @ref simplemc-accs-stats-covar). Its imaginary part \f$ ([s_{\mathbf{XY}}^2]^T -
     * s_{\mathbf{XY}}^2) \f$ is anti-Hermitian, so the diagonal is real but the off-diagonal entries
     * are in general complex. For scalar complex samples the result is the real-valued \f$
     * s_{\mathbf{f}\mathbf{f}}^2 = \mathrm{Var}(\mathrm{Re}\,f) + \mathrm{Var}(\mathrm{Im}\,f) \f$
     * and matches variance().
     *
     * @return Jackknife covariance estimate \f$ s_{\mathbf{f}\mathbf{f}}^2 \f$.
     */
    [[nodiscard]] auto covariance() const {
        const auto nm1 = static_cast<double>(acc_.count() - 1);
        const auto scale = nm1 * nm1;
        if constexpr (sample_scalar<sample_type>) {
            return scale * acc_.covariance();
        } else {
            return (scale * acc_.covariance()).eval();
        }
    }

    /**
     * @brief Calculate the jackknife bias estimate \f$ b_{\text{JK}} \f$.
     *
     * @details It computes \f$ b_{\text{JK}} = (N-1) \left( \overline{f}_{\text{JK}} -
     * f_{\text{naive}} \right) \f$.
     *
     * @return Jackknife bias estimate \f$ b_{\text{JK}} \f$.
     */
    [[nodiscard]] auto bias() const {
        const auto nm1 = static_cast<double>(acc_.count() - 1);
        const auto jk_mean = acc_.mean();
        if constexpr (sample_scalar<sample_type>) {
            return nm1 * (jk_mean - f_naive_);
        } else {
            return (nm1 * (jk_mean - f_naive_)).eval();
        }
    }

    /**
     * @brief Calculate the bias-corrected mean \f$ \overline{f}_{\text{corr}} \f$.
     *
     * @details It computes \f$ \overline{f}_{\text{corr}} = N f_{\text{naive}} - (N - 1)
     * \overline{f}_{\text{JK}} = f_{\text{naive}} - b_{\text{JK}} \f$.
     *
     * @return Bias-corrected mean \f$ \overline{f}_{\text{corr}} \f$.
     */
    [[nodiscard]] auto bias_corrected_mean() const {
        const auto n = static_cast<double>(acc_.count());
        const auto nm1 = n - 1.0;
        const auto jk_mean = acc_.mean();
        if constexpr (sample_scalar<sample_type>) {
            return n * f_naive_ - nm1 * jk_mean;
        } else {
            return (n * f_naive_ - nm1 * jk_mean).eval();
        }
    }

private:
    sample_type f_naive_;
    acc_type acc_;
};

/** @} */

namespace detail {

// Core jackknife implementation. Accumulates delete-one jackknife estimates in a covar_acc and
// returns a jackknife_result.
template <varalg Alg, sample_range... Rs>
    requires(sizeof...(Rs) > 0)
auto jackknife_make_acc(auto&& f, Rs&&... rgs) { // NOLINT (ranges need not be forwarded)
    // check that the ranges all have the same size > 1 (= number of samples)
    auto rgs_tup = std::tie(rgs...);
    auto const count = ranges::size(std::get<0>(rgs_tup));
    if (((ranges::size(rgs) != count) || ...)) {
        throw simplemc_exception("Random sample ranges have different sizes");
    }
    if (count < 2) {
        throw simplemc_exception("Random sample ranges contain < 2 samples");
    }

    // get the mean of each range and store it in a tuple
    auto mean_tup = [&rgs_tup]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(make_mean_acc(std::get<Is>(rgs_tup)).mean()...);
    }(std::make_index_sequence<sizeof...(rgs)> {});

    // compute the original estimate by applying f to the per-observable means
    auto original_mean = std::apply(f, mean_tup);

    // get the jackknife samples for each range and store them in a tuple (of ranges)
    // note: using sample_type R(...) forces evaluation of Eigen expression templates
    auto const n = static_cast<double>(count);
    auto jk_tup = [n, &rgs_tup, &mean_tup]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(
            ranges::transform_view(std::get<Is>(rgs_tup), [m = std::get<Is>(mean_tup), n](auto const& x) {
                using R = ranges::range_value_t<std::remove_cvref_t<decltype(std::get<Is>(rgs_tup))>>;
                return R((n * m - x) / (n - 1));
            })...);
    }(std::make_index_sequence<sizeof...(rgs)> {});

    // get a zipped view of the jackknife samples (tuple of ranges --> range of tuples)
    auto jk_zipped = std::apply([](auto&&... jks) { return ranges::zip_view(jks...); }, jk_tup);

    // accumulate jackknife samples in a covariance accumulator and return the result
    using result_type = std::remove_cvref_t<decltype(original_mean)>;
    auto apply_f = [f](const auto& tup) -> result_type { return std::apply(f, tup); };
    auto acc = make_covar_acc<Alg>(ranges::transform_view(jk_zipped, apply_f));
    return jackknife_result(std::move(original_mean), std::move(acc));
}

} // namespace detail

/**
 * @addtogroup simplemc-accs-resampling
 * @{
 */

/**
 * @brief Perform jackknife resampling for one or more ranges of samples.
 *
 * @anchor jackknife_range_overload
 * @details Given \f$ K \geq 1 \f$ sample ranges \f$ S_{\mathbf{Z}_1}, \ldots, S_{\mathbf{Z}_K} \f$ of
 * equal size \f$ N \f$ and a \f$ K \f$-variate transformation function \f$ f \f$, we first generate
 * the \f$ N \f$ delete-one jackknife estimates \f$ f_{[j]}^{(N-1)} \f$, before accumulating them in a
 * simplemc::covar_acc.
 *
 * The accumulator together with the naive estimate \f$ f_{\text{naive}} \f$ is returned in a
 * simplemc::jackknife_result.
 *
 * It throws a simplemc::simplemc_exception if the ranges have different sizes or contain fewer than 2
 * samples.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type of \f$ f \f$.
 * @tparam Rs simplemc::sample_range types.
 * @param f Transformation function \f$ f \f$.
 * @param rgs One or more ranges \f$ S_{\mathbf{Z}_i} \f$ containing the data samples.
 * @return simplemc::jackknife_result wrapping a simplemc::covar_acc of the jackknife estimates.
 */
template <varalg A = varalg::welford, typename F, sample_range... Rs>
    requires(sizeof...(Rs) >= 1 && std::invocable<F, ranges::range_value_t<Rs>...>)
[[nodiscard]] auto jackknife(F&& f, Rs&&... rgs) {
    return detail::jackknife_make_acc<A>(std::forward<F>(f), std::forward<Rs>(rgs)...);
}

/**
 * @brief Perform jackknife resampling for one or more simplemc::batch_acc objects.
 *
 * @anchor jackknife_batch_overload
 * @details Given \f$ K \geq 1 \f$ batch accumulators with equal numbers \f$ N \f$ of full batches
 * and a \f$ K \f$-variate transformation function \f$ f \f$, we treat the \f$ N \f$ batch means of
 * each accumulator as the input samples and generate the \f$ N \f$ delete-one-batch jackknife
 * estimates \f$ f_{[j]}^{(N-1)} \f$, before accumulating them in a simplemc::covar_acc.
 *
 * The accumulator together with the naive estimate \f$ f_{\text{naive}} \f$ is returned in a
 * simplemc::jackknife_result.
 *
 * It throws a simplemc::simplemc_exception if any batch accumulator has no full batches or if the
 * batch accumulators have different numbers of full batches.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type of \f$ f \f$.
 * @tparam Bs simplemc::batch_accumulator types.
 * @param f Transformation function \f$ f \f$.
 * @param baccs One or more batch accumulators containing the data for each observable.
 * @return simplemc::jackknife_result wrapping a simplemc::covar_acc of the jackknife estimates.
 */
template <varalg A = varalg::welford, typename F, batch_accumulator... Bs>
    requires(sizeof...(Bs) >= 1 && std::invocable<F, typename Bs::sample_type...>)
[[nodiscard]] auto jackknife(F&& f, const Bs&... baccs) {
    // get the batch means of each accumulator and store them in a tuple as a vector of means
    auto extract_means = []<batch_accumulator B>(const B& bacc) {
        auto bs = bacc.batches();
        if (bs.empty()) {
            throw simplemc_exception("batch accumulator has no full batches");
        }
        std::vector<typename B::sample_type> means;
        means.reserve(bs.size());
        for (const auto& b : bs) {
            means.push_back(b.mean());
        }
        return means;
    };
    auto means_tup = std::make_tuple(extract_means(baccs)...);

    // check that all batch accumulators have the same number of full batches
    const auto n0 = std::get<0>(means_tup).size();
    bool sizes_match = std::apply([n0](const auto&... ms) { return ((ms.size() == n0) && ...); }, means_tup);
    if (!sizes_match) {
        throw simplemc_exception("batch accumulators have different numbers of full batches");
    }

    // accumulate jackknife samples in a covariance accumulator and return the result
    return std::apply(
        [&f](auto&... means) { return detail::jackknife_make_acc<A>(std::forward<F>(f), means...); }, means_tup);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_JACKKNIFE_HPP
