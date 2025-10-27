/**
 * @file
 * @brief Jackknife resampling.
 */

#ifndef SIMPLEMC_ACCS_JACKKNIFE_HPP
#define SIMPLEMC_ACCS_JACKKNIFE_HPP

#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cstddef>
#include <tuple>
#include <utility>

namespace simplemc {

namespace detail {

// Accumulate jackknife samples in either a var_acc or covar_acc.
template <bool do_covar, typename F, random_sample_range... Rs>
    requires(sizeof...(Rs) > 0)
auto jackknife_make_acc(auto&& f, Rs&&... rgs) { // NOLINT (ranges need not be forwarded)
    // check that the ranges all have the same size > 1 (= number of samples)
    auto rgs_tup = std::tie(rgs...);
    auto const count = ranges::size(std::get<0>(rgs_tup));
    if (((ranges::size(rgs) != count) || ...)) {
        throw simplemc_exception("Random sample ranges have different sizes", "detail::jackknife_make_acc");
    }
    if (count < 2) {
        throw simplemc_exception("Random sample ranges contain < 2 samples", "detail::jackknife_make_acc");
    }

    // get the mean of each range and store it in a tuple
    auto mean_tup = [&rgs_tup]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(make_mean_acc(std::get<Is>(rgs_tup)).mean()...);
    }(std::make_index_sequence<sizeof...(rgs)> {});

    // get the jackknife samples for each range and store them in a tuple (of ranges)
    auto jk_tup = [count, &rgs_tup, &mean_tup]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(ranges::transform_view(std::get<Is>(rgs_tup),
            [m = std::get<Is>(mean_tup), count](auto const& x) { return (count * m - x) / (count - 1); })...);
    }(std::make_index_sequence<sizeof...(rgs)> {});

    // get a zipped view of the jackknife samples (tuple of ranges --> range of tuples)
    auto jk_zipped = std::apply([](auto&&... jks) { return ranges::zip_view(jks...); }, jk_tup);

    // accumulate jackknife samples in a (co)variance accumulator and return it
    auto apply_f = [f](const auto& tup) { return std::apply(f, tup); };
    if constexpr (do_covar) {
        return std::make_tuple(mean_tup, make_covar_acc(ranges::transform_view(jk_zipped, apply_f)));
    } else {
        return std::make_tuple(mean_tup, make_var_acc(ranges::transform_view(jk_zipped, apply_f)));
    }
}

} // namespace detail

/**
 * @addtogroup simplemc-accs-resampling
 * @{
 */



/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_JACKKNIFE_HPP
