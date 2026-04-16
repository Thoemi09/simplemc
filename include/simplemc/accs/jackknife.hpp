/**
 * @file
 * @brief Jackknife resampling for accumulators.
 */

#ifndef SIMPLEMC_ACCS_JACKKNIFE_HPP
#define SIMPLEMC_ACCS_JACKKNIFE_HPP

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cstddef>
#include <functional>
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
 * \overline{\mathbf{z}_2}^{(N)}, \dots) \f$ and a variance or covariance accumulator of \f$ N \f$
 * jackknife samples \f$ f_{[i]}^{(N-1)} \f$.
 *
 * It provides convenience methods that apply the correct jackknife scaling factors to obtain
 * estimates for the standard error, variance and covariance matrix of some function \f$ f \f$ of
 * expectation values. Furthermore, it allows to calculate the jackknife bias estimate and the
 * bias-corrected estimate \f$ \overline{f}_{\text{corr}} \f$.
 *
 * See @ref simplemc-accs-resampling for more details.
 *
 * @tparam A Accumulator type satisfying simplemc::variance_accumulator.
 */
template <variance_accumulator A>
class jackknife_result {
public:
    /**
     * @brief Type of the wrapped accumulator.
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
     * @return simplemc::varalg tag of the wrapped accumulator.
     */
    static constexpr auto varalg() noexcept { return A::varalg(); }

public:
    /**
     * @brief Construct a jackknife result.
     *
     * @param f_naive Naive estimate \f$ f_{\text{naive}} \f$.
     * @param acc Variance/covariance accumulator of jackknife samples.
     */
    jackknife_result(sample_type f_naive, A acc) : f_naive_(std::move(f_naive)), acc_(std::move(acc)) {}

    /**
     * @brief Get the wrapped accumulator.
     *
     * @return Const reference to the stored variance/covariance accumulator.
     */
    [[nodiscard]] const acc_type& accumulator() const noexcept { return acc_; }

    /**
     * @brief Get the number of accumulated samples \f$ N \f$.
     *
     * @return Number of jackknife samples.
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
     * @brief Get the mean \f$ \overline{f}_{\text{JK}} \f$ of the jackknife estimates.
     *
     * @return Mean of the jackknife estimates.
     */
    [[nodiscard]] auto mean() const { return acc_.mean(); }

    /**
     * @brief Calculate the jackknife variance estimate \f$ s_f^2 \f$.
     *
     * @details It computes \f$ s_f^2 = (N-1)^{2} \, s_{\overline{f}}^{2} \f$, where \f$
     * s_{\overline{f}}^{2} \f$ is the variance of the mean of the jackknife estimates as returned by
     * the wrapped accumulator.
     *
     * @return Jackknife variance estimate \f$ s_f^2 \f$.
     */
    [[nodiscard]] auto variance() const {
        const auto nm1 = static_cast<double>(acc_.count() - 1);
        const auto scale = nm1 * nm1;
        if constexpr (sample_scalar<sample_type>) {
            return scale * acc_.variance();
        } else {
            return (scale * acc_.variance()).eval();
        }
    }

    /**
     * @brief Calculate the jackknife covariance estimate \f$ s_{\mathbf{f}\mathbf{f}}^2 \f$.
     *
     * @details It computes \f$ s_{\mathbf{f}\mathbf{f}}^2 = (N-1)^{2} \, s_{\overline{\mathbf{f}}
     * \\overline{\mathbf{f}}}^{2} \f$, where \f$ s_{\overline{\mathbf{f}}\overline{\mathbf{f}}}^{2}
     * \f$ is the sample covariance matrix of the mean of the jackknife estimates.
     *
     * @note Only available when the wrapped accumulator satisfies simplemc::covariance_accumulator.
     *
     * @return Jackknife covariance estimate \f$ s_{\mathbf{f}\mathbf{f}}^2 \f$.
     */
    [[nodiscard]] auto covariance() const
        requires covariance_accumulator<acc_type>
    {
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
     * @details It computes \f$ b_{\text{JK}} = (N-1)(\overline{f}_{\text{JK}} -
     * f(\overline{\mathbf{z}_1}^{(N)}, \overline{\mathbf{z}_2}^{(N)}, \dots)) \f$.
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
     * @details It computes \f$ \overline{f}_{\text{corr}} = N f(\overline{\mathbf{z}_1}^{(N)}, \dots)
     * - (N-1) \overline{f}_{\text{JK}} \f$.
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

// Trait to detect batch_acc types.
template <typename T>
struct is_batch_acc : std::false_type {};

template <sample_type T, varalg A>
struct is_batch_acc<batch_acc<T, A>> : std::true_type {};

// Extract the sample_type from a batch_acc type.
template <typename B>
using batch_sample_t = typename std::remove_cvref_t<B>::sample_type;

// Extract batch means from a batch_acc as a std::vector.
template <sample_type T, varalg A>
[[nodiscard]] auto batch_means(const batch_acc<T, A>& bacc) {
    auto bs = bacc.batches();
    if (bs.empty()) {
        throw simplemc_exception("batch_acc has no full batches");
    }
    std::vector<T> means;
    means.reserve(bs.size());
    for (const auto& b : bs) {
        means.push_back(b.mean());
    }
    return means;
}

// Core jackknife implementation. Accumulates delete-one jackknife estimates in a var_acc
// (do_covar=false) or covar_acc (do_covar=true) and returns a jackknife_result.
template <bool do_covar, varalg Alg, sample_range... Rs>
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

    // accumulate jackknife samples in a (co)variance accumulator and return the result
    using result_type = std::remove_cvref_t<decltype(original_mean)>;
    auto apply_f = [f](const auto& tup) -> result_type { return std::apply(f, tup); };
    if constexpr (do_covar) {
        auto acc = make_covar_acc<Alg>(ranges::transform_view(jk_zipped, apply_f));
        return jackknife_result(std::move(original_mean), std::move(acc));
    } else {
        auto acc = make_var_acc<Alg>(ranges::transform_view(jk_zipped, apply_f));
        return jackknife_result(std::move(original_mean), std::move(acc));
    }
}

} // namespace detail

/**
 * @addtogroup simplemc-accs-resampling
 * @{
 */

// ================================================================================================
// Range-based jackknife
// ================================================================================================

/**
 * @brief Compute jackknife variance from a single range of samples.
 *
 * @details Generates \f$ N \f$ delete-one jackknife estimates
 * \f$ f_{[j]}^{(N-1)} = f\!\bigl((N \overline{\mathbf{z}}^{(N)} - \mathbf{z}^{(j)}) /
 * (N-1)\bigr) \f$ and accumulates them in a simplemc::var_acc. The result is returned as a
 * simplemc::jackknife_result.
 *
 * It throws a simplemc::simplemc_exception if the range contains fewer than 2 samples.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam R simplemc::sample_range type.
 * @tparam F Callable type applied to each jackknife sample (defaults to `std::identity`).
 * @param rg Range containing the data samples \f$ \{\mathbf{z}^{(j)}\} \f$.
 * @param f Transformation function \f$ f \f$ applied to each jackknife sample.
 * @return simplemc::jackknife_result wrapping a simplemc::var_acc of the jackknife estimates.
 */
template <varalg A = varalg::welford, sample_range R, typename F = std::identity>
    requires std::invocable<F, ranges::range_value_t<R>>
[[nodiscard]] auto make_jackknife_var_acc(R&& rg, F&& f = {}) {
    return detail::jackknife_make_acc<false, A>(std::forward<F>(f), std::forward<R>(rg));
}

/**
 * @brief Compute jackknife covariance from a single range of samples.
 *
 * @details Generates \f$ N \f$ delete-one jackknife estimates
 * \f$ f_{[j]}^{(N-1)} = f\!\bigl((N \overline{\mathbf{z}}^{(N)} - \mathbf{z}^{(j)}) /
 * (N-1)\bigr) \f$ and accumulates them in a simplemc::covar_acc. The result is returned as a
 * simplemc::jackknife_result.
 *
 * It throws a simplemc::simplemc_exception if the range contains fewer than 2 samples.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam R simplemc::sample_range type.
 * @tparam F Callable type applied to each jackknife sample (defaults to `std::identity`).
 * @param rg Range containing the data samples \f$ \{\mathbf{z}^{(j)}\} \f$.
 * @param f Transformation function \f$ f \f$ applied to each jackknife sample.
 * @return simplemc::jackknife_result wrapping a simplemc::covar_acc of the jackknife estimates.
 */
template <varalg A = varalg::welford, sample_range R, typename F = std::identity>
    requires std::invocable<F, ranges::range_value_t<R>>
[[nodiscard]] auto make_jackknife_covar_acc(R&& rg, F&& f = {}) {
    return detail::jackknife_make_acc<true, A>(std::forward<F>(f), std::forward<R>(rg));
}

/**
 * @brief Compute jackknife variance from multiple ranges of samples.
 *
 * @details For \f$ K > 1 \f$ observables with sample ranges
 * \f$ S_{\mathbf{Z}_1}, \ldots, S_{\mathbf{Z}_K} \f$ of equal size \f$ N \f$, generates
 * \f$ N \f$ delete-one jackknife estimates
 * \f$ f_{[j]}^{(N-1)} = f\!\bigl(\mathbf{z}_{1[j]}^{(N-1)}, \ldots,
 * \mathbf{z}_{K[j]}^{(N-1)}\bigr) \f$, where
 * \f$ \mathbf{z}_{k[j]}^{(N-1)} = (N \overline{\mathbf{z}_k}^{(N)} -
 * \mathbf{z}_k^{(j)}) / (N-1) \f$. The estimates are accumulated in a simplemc::var_acc.
 *
 * It throws a simplemc::simplemc_exception if the ranges have different sizes or contain fewer
 * than 2 samples.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type that combines the observables.
 * @tparam Rs simplemc::sample_range types.
 * @param f Transformation function \f$ f \f$ applied to the tuple of jackknife samples.
 * @param rgs Ranges containing the data samples.
 * @return simplemc::jackknife_result wrapping a simplemc::var_acc of the jackknife estimates.
 */
template <varalg A = varalg::welford, typename F, sample_range... Rs>
    requires(sizeof...(Rs) > 1 && std::invocable<F, ranges::range_value_t<Rs>...>)
[[nodiscard]] auto make_jackknife_var_acc(F&& f, Rs&&... rgs) {
    return detail::jackknife_make_acc<false, A>(std::forward<F>(f), std::forward<Rs>(rgs)...);
}

/**
 * @brief Compute jackknife covariance from multiple ranges of samples.
 *
 * @details For \f$ K > 1 \f$ observables with sample ranges
 * \f$ S_{\mathbf{Z}_1}, \ldots, S_{\mathbf{Z}_K} \f$ of equal size \f$ N \f$, generates
 * \f$ N \f$ delete-one jackknife estimates
 * \f$ f_{[j]}^{(N-1)} = f\!\bigl(\mathbf{z}_{1[j]}^{(N-1)}, \ldots,
 * \mathbf{z}_{K[j]}^{(N-1)}\bigr) \f$, where
 * \f$ \mathbf{z}_{k[j]}^{(N-1)} = (N \overline{\mathbf{z}_k}^{(N)} -
 * \mathbf{z}_k^{(j)}) / (N-1) \f$. The estimates are accumulated in a
 * simplemc::covar_acc.
 *
 * It throws a simplemc::simplemc_exception if the ranges have different sizes or contain fewer
 * than 2 samples.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type that combines the observables.
 * @tparam Rs simplemc::sample_range types.
 * @param f Transformation function \f$ f \f$ applied to the tuple of jackknife samples.
 * @param rgs Ranges containing the data samples.
 * @return simplemc::jackknife_result wrapping a simplemc::covar_acc of the jackknife estimates.
 */
template <varalg A = varalg::welford, typename F, sample_range... Rs>
    requires(sizeof...(Rs) > 1 && std::invocable<F, ranges::range_value_t<Rs>...>)
[[nodiscard]] auto make_jackknife_covar_acc(F&& f, Rs&&... rgs) {
    return detail::jackknife_make_acc<true, A>(std::forward<F>(f), std::forward<Rs>(rgs)...);
}

// ================================================================================================
// batch_acc-based jackknife
// ================================================================================================

/**
 * @brief Compute jackknife variance from a simplemc::batch_acc.
 *
 * @details Extracts the full batch means from the given simplemc::batch_acc and computes the
 * delete-one-batch jackknife. Each batch mean serves as one jackknife sample. The transformation
 * function \f$ f \f$ is applied to each jackknife estimate.
 *
 * It throws a simplemc::simplemc_exception if the simplemc::batch_acc has no full batches.
 *
 * @tparam Alg simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type applied to each jackknife sample (defaults to `std::identity`).
 * @tparam T simplemc::sample_type of the batch accumulator.
 * @tparam A simplemc::varalg of the batch accumulator.
 * @param bacc Batch accumulator containing the data.
 * @param f Transformation function \f$ f \f$ applied to each jackknife sample.
 * @return simplemc::jackknife_result wrapping a simplemc::var_acc of the jackknife estimates.
 */
template <varalg Alg = varalg::welford, typename F = std::identity, sample_type T, varalg A>
    requires std::invocable<F, T>
[[nodiscard]] auto make_jackknife_var_acc(const batch_acc<T, A>& bacc, F&& f = {}) {
    auto means = detail::batch_means(bacc);
    return detail::jackknife_make_acc<false, Alg>(std::forward<F>(f), means);
}

/**
 * @brief Compute jackknife covariance from a simplemc::batch_acc.
 *
 * @details Extracts the full batch means from the given simplemc::batch_acc and computes the
 * delete-one-batch jackknife. Each batch mean serves as one jackknife sample. The transformation
 * function \f$ f \f$ is applied to each jackknife estimate.
 *
 * It throws a simplemc::simplemc_exception if the simplemc::batch_acc has no full batches.
 *
 * @tparam Alg simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type applied to each jackknife sample (defaults to `std::identity`).
 * @tparam T simplemc::sample_type of the batch accumulator.
 * @tparam A simplemc::varalg of the batch accumulator.
 * @param bacc Batch accumulator containing the data.
 * @param f Transformation function \f$ f \f$ applied to each jackknife sample.
 * @return simplemc::jackknife_result wrapping a simplemc::covar_acc of the jackknife estimates.
 */
template <varalg Alg = varalg::welford, typename F = std::identity, sample_type T, varalg A>
    requires std::invocable<F, T>
[[nodiscard]] auto make_jackknife_covar_acc(const batch_acc<T, A>& bacc, F&& f = {}) {
    auto means = detail::batch_means(bacc);
    return detail::jackknife_make_acc<true, Alg>(std::forward<F>(f), means);
}

/**
 * @brief Compute jackknife variance from multiple simplemc::batch_acc objects.
 *
 * @details Extracts batch means from each simplemc::batch_acc and computes the multi-observable
 * delete-one-batch jackknife. The transformation function \f$ f \f$ is applied to each tuple of
 * jackknife estimates.
 *
 * It throws a simplemc::simplemc_exception if any simplemc::batch_acc has no full batches or if
 * the batch accumulators have different numbers of full batches.
 *
 * @tparam Alg simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type that combines the observables.
 * @tparam Bs simplemc::batch_acc types.
 * @param f Transformation function \f$ f \f$ applied to the tuple of jackknife samples.
 * @param baccs Batch accumulators containing the data for each observable.
 * @return simplemc::jackknife_result wrapping a simplemc::var_acc of the jackknife estimates.
 */
template <varalg Alg = varalg::welford, typename F, typename... Bs>
    requires(sizeof...(Bs) > 1 && (detail::is_batch_acc<std::remove_cvref_t<Bs>>::value && ...) &&
        std::invocable<F, detail::batch_sample_t<Bs>...>)
[[nodiscard]] auto make_jackknife_var_acc(F&& f, const Bs&... baccs) {
    auto means_tup = std::make_tuple(detail::batch_means(baccs)...);
    return std::apply(
        [&f](auto&... means) { return detail::jackknife_make_acc<false, Alg>(std::forward<F>(f), means...); },
        means_tup);
}

/**
 * @brief Compute jackknife covariance from multiple simplemc::batch_acc objects.
 *
 * @details Extracts batch means from each simplemc::batch_acc and computes the multi-observable
 * delete-one-batch jackknife. The transformation function \f$ f \f$ is applied to each tuple of
 * jackknife estimates.
 *
 * It throws a simplemc::simplemc_exception if any simplemc::batch_acc has no full batches or if
 * the batch accumulators have different numbers of full batches.
 *
 * @tparam Alg simplemc::varalg algorithm used to accumulate the jackknife estimates.
 * @tparam F Callable type that combines the observables.
 * @tparam Bs simplemc::batch_acc types.
 * @param f Transformation function \f$ f \f$ applied to the tuple of jackknife samples.
 * @param baccs Batch accumulators containing the data for each observable.
 * @return simplemc::jackknife_result wrapping a simplemc::covar_acc of the jackknife estimates.
 */
template <varalg Alg = varalg::welford, typename F, typename... Bs>
    requires(sizeof...(Bs) > 1 && (detail::is_batch_acc<std::remove_cvref_t<Bs>>::value && ...) &&
        std::invocable<F, detail::batch_sample_t<Bs>...>)
[[nodiscard]] auto make_jackknife_covar_acc(F&& f, const Bs&... baccs) {
    auto means_tup = std::make_tuple(detail::batch_means(baccs)...);
    return std::apply(
        [&f](auto&... means) { return detail::jackknife_make_acc<true, Alg>(std::forward<F>(f), means...); },
        means_tup);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_JACKKNIFE_HPP
