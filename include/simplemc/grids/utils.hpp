/**
 * @file
 * @brief Utility functions for @ref simplemc-grids-1d and @ref simplemc-grids-nd.
 */

#ifndef SIMPLEMC_GRIDS_UTILS_HPP
#define SIMPLEMC_GRIDS_UTILS_HPP

#include <simplemc/grids/concepts.hpp>
#include <simplemc/utils/nd_indexing.hpp>
#include <simplemc/utils/ranges.hpp>

#include <tuple>

namespace simplemc {

/**
 * @ingroup simplemc-grids-1d
 * @brief Find a subrange of consecutive grid points of a 1-dimensional grid \f$ g \f$ which contains
 * a given value \f$ x \f$.
 *
 * @details The subrange consists of \f$ m \f$ consecutive grid points and is chosen such that the
 * given value \f$ x \in \mathrm{R} [a, b] \f$ lies, as far as possible, in the middle of the interval
 * - \f$ [g(i), g(i+m-1)] \f$ for increasing grids or
 * - \f$ [g(i+m-1), g(i)] \f$ for decreasing grids.
 *
 * It first finds the index \f$ i \f$ of the bin \f$ b_i \f$ which contains \f$ x \f$ and then
 * forwards the result to simplemc::integer_subrange.
 *
 * @tparam G1 simplemc::grid_1d type.
 * @param grid 1-dimensional grid \f$ g \f$.
 * @param m Size of the desired subrange.
 * @param x Some value \f$ x \in [a, b] \f$.
 * @return Index of bin \f$ i \f$ such that \f$ x \f$ lies in the middle of \f$ [g(i), g(i+m-1)] \f$
 * (or \f$ [g(i+m-1), g(i)] \f$ for decreasing grids).
 */
template <grid_1d G1>
[[nodiscard]] constexpr auto index_subrange(const G1& grid, long m, typename G1::value_type x) {
    return integer_subrange(grid.index(x), grid.size(), m);
}

/**
 * @ingroup simplemc-grids-nd
 * @brief Find a subvolume of neighboring bins of an N-dimensional grid \f$ g \f$ which contains a
 * given point \f$ \mathbf{x} \f$.
 *
 * @details It calls simplemc::index_subrange for each 1-dimensional grid.
 *
 * @tparam GN simplemc::grid_nd type.
 * @param grid N-dimensional grid \f$ g \f$.
 * @param m Size of the desired subrange.
 * @param x Some point \f$ \mathbf{x} = (x_1, \dots, x_N) \in \mathrm{R} \f$.
 * @return Index array \f$ \mathbf{i} = (i_1, \dots, i_N) \f$ of the bin with the smallest indices in
 * the subvolume.
 */
template <grid_nd GN>
[[nodiscard]] constexpr auto index_subrange(const GN& grid, long m, const typename GN::value_type& x) {
    return std::apply([&grid, m](const auto&... args) { return index_subrange(grid, m, args...); }, x);
}

/**
 * @ingroup simplemc-grids-nd
 * @brief Find a subvolume of neighboring bins of an N-dimensional grid \f$ g \f$ which contains a
 * given point \f$ \mathbf{x} \f$.
 *
 * @details It calls simplemc::index_subrange for each 1-dimensional grid.
 *
 * @tparam GN simplemc::grid_nd type
 * @tparam Vals Value types of 1-dimensional grids.
 * @param grid N-dimensional grid \f$ g \f$.
 * @param m Size of the desired subrange.
 * @param xs 1-dimensional points \f$ x_1, \dots, x_N \f$.
 * @return Index array \f$ \mathbf{i} = (i_1, \dots, i_N) \f$ of the bin with the smallest indices in
 * the subvolume.
 */
template <grid_nd GN, typename... Vals>
[[nodiscard]] constexpr auto index_subrange(const GN& grid, long m, Vals... xs) {
    return std::apply([m, xs...](const auto&... gs) { return typename GN::size_type { index_subrange(gs, m, xs)... }; },
        grid.grids());
}

/**
 * @ingroup simplemc-grids-1d
 * @brief Get a lazy view on the grid points of a 1-dimensional grid \f$ g \f$.
 *
 * @details
 * @code{.cpp}
 * // create a linear grid of size 5 on [0, 1]
 * simplemc::linear_grid lg(0.0, 1.0, 5);
 *
 * // print the grid points as a view
 * fmt::println("{}", simplemc::grid_view(lg));
 * @endcode
 *
 * Output:
 *
 * ```
 * [0, 0.25, 0.5, 0.75, 1]
 * ```
 *
 * @note All @ref simplemc-grids-1d satisfy the `std::ranges::random_access_range` concept. It might
 * be easier to use the grid directly instead of using this view.
 *
 * @tparam G1 simplemc::grid_1d type.
 * @param grid 1-dimensional grid \f$ g \f$.
 * @return View on the grid points.
 */
template <grid_1d G1>
[[nodiscard]] constexpr auto grid_view(const G1& grid) {
    using size_type = typename G1::size_type;
    return ranges::iota_view<size_type, size_type>(0, grid.size()) |
        ranges::views::transform([&grid](auto idx) { return grid.at(idx); });
}

/**
 * @ingroup simplemc-grids-nd
 * @brief Get a lazy view on the grid points of an N-dimensional grid \f$ g \f$.
 *
 * @details The grid is traversed in row-major (C) order, i.e. the last dimension is the fastest
 * varying one and the first dimension is the slowest varying one (see simplemc::row_major).
 *
 * @code{.cpp}
 * // create a 2d grid
 * simplemc::linear_grid lg { 0.0, 1.0, 3 };
 * simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
 * simplemc::nd_grid grid_2d { lg, pg };
 *
 * // print the grid points as a view
 * fmt::println("{}", simplemc::grid_view(grid_2d));
 * @endcode
 *
 * Output:
 *
 * ```
 * [[0, 0], [0, 0.125], [0, 0.5], [0.5, 0], [0.5, 0.125], [0.5, 0.5], [1, 0], [1, 0.125], [1, 0.5]]
 * ```
 *
 * @note All simplemc::grid_nd types satisfy the `std::ranges::random_access_range` concept. It might
 * be easier to use the grid directly instead of using this view.
 *
 * @tparam GN simplemc::grid_nd type.
 * @param grid N-dimensional grid \f$ g \f$.
 * @return View on the grid points.
 */
template <grid_nd GN>
[[nodiscard]] constexpr auto grid_view(const GN& grid) {
    return std::apply(
        [](const auto&... gs) {
            return ranges::views::cartesian_product(grid_view(gs)...) | ranges::views::transform([](const auto& tup) {
                return std::apply([](const auto&... vals) { return typename GN::value_type { vals... }; }, tup);
            });
        },
        grid.grids());
}

/**
 * @ingroup simplemc-grids-1d
 * @brief Get a lazy view on the bin centers of a 1-dimensional grid \f$ g \f$.
 *
 * @details
 * @code{.cpp}
 * // create a linear grid of size 5 on [0, 1]
 * simplemc::linear_grid lg(0.0, 1.0, 5);
 *
 * // print the bin centers as a view
 * fmt::println("{}", simplemc::bin_center_view(lg));
 * @endcode
 *
 * Output:
 *
 * ```
 * [0.125, 0.375, 0.625, 0.875]
 * ```
 *
 * @tparam G1 simplemc::grid_1d type.
 * @param grid 1-dimensional grid \f$ g \f$.
 * @return View on the bin centers.
 */
template <grid_1d G1>
[[nodiscard]] constexpr auto bin_center_view(const G1& grid) {
    using size_type = typename G1::size_type;
    return ranges::iota_view<size_type, size_type>(0, grid.size() - 1) |
        ranges::views::transform([&grid](auto idx) { return grid.center(idx); });
}

/**
 * @ingroup simplemc-grids-nd
 * @brief Get a lazy view on the bin centers of an N-dimensional grid \f$ g \f$.
 *
 * @details The grid is traversed in row-major (C) order, i.e. the last dimension is the fastest
 * varying one and the first dimension is the slowest varying one (see simplemc::row_major).
 *
 * @code{.cpp}
 * // create a 2d grid
 * simplemc::linear_grid lg { 0.0, 1.0, 3 };
 * simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
 * simplemc::nd_grid grid_2d { lg, pg };
 *
 * // print the bin centers as a view
 * fmt::println("{}", simplemc::bin_center_view(grid_2d));
 * @endcode
 *
 * Output:
 *
 * ```
 * [[0.25, 0.0625], [0.25, 0.3125], [0.75, 0.0625], [0.75, 0.3125]]
 * ```
 *
 * @tparam GN simplemc::grid_nd type.
 * @param grid N-dimensional grid \f$ g \f$.
 * @return View on the bin centers.
 */
template <grid_nd GN>
[[nodiscard]] constexpr auto bin_center_view(const GN& grid) {
    return std::apply(
        [](const auto&... gs) {
            return ranges::views::cartesian_product(bin_center_view(gs)...) |
                ranges::views::transform([](const auto& tup) {
                    return std::apply([](const auto&... vals) { return typename GN::value_type { vals... }; }, tup);
                });
        },
        grid.grids());
}

/**
 * @ingroup simplemc-grids-1d
 * @brief Get a lazy view on the bin volumes of a 1-dimensional grid \f$ g \f$.
 *
 * @details
 * @code{.cpp}
 * // create a linear grid of size 5 on [0, 1]
 * simplemc::linear_grid lg(0.0, 1.0, 5);
 *
 * // print the bin volumes as a view
 * fmt::println("{}", simplemc::bin_volume_view(lg));
 * @endcode
 *
 * Output:
 *
 * ```
 * [0.25, 0.25, 0.25, 0.25]
 * ```
 *
 * @tparam G1 simplemc::grid_1d type.
 * @param grid 1-dimensional grid \f$ g \f$.
 * @return View on the bin volumes.
 */
template <grid_1d G1>
[[nodiscard]] constexpr auto bin_volume_view(const G1& grid) {
    using size_type = typename G1::size_type;
    return ranges::iota_view<size_type, size_type>(0, grid.size() - 1) |
        ranges::views::transform([&grid](auto idx) { return grid.volume(idx); });
}

/**
 * @ingroup simplemc-grids-nd
 * @brief Get a lazy view on the bin volumes of an N-dimensional grid \f$ g \f$.
 *
 * @details The grid is traversed in row-major (C) order, i.e. the last dimension is the fastest
 * varying one and the first dimension is the slowest varying one (see simplemc::row_major).
 *
 * @code{.cpp}
 * // create a 2d grid
 * simplemc::linear_grid lg { 0.0, 1.0, 3 };
 * simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
 * simplemc::nd_grid grid_2d { lg, pg };
 *
 * // print the bin volumes as a view
 * fmt::println("{}", simplemc::bin_volume_view(grid_2d));
 * @endcode
 *
 * Output:
 *
 * ```
 * [0.0625, 0.1875, 0.0625, 0.1875]
 * ```
 *
 * @tparam GN simplemc::grid_nd type.
 * @param grid N-dimensional grid \f$ g \f$.
 * @return View on the bin volumes.
 */
template <grid_nd GN>
[[nodiscard]] constexpr auto bin_volume_view(const GN& grid) {
    return std::apply(
        [](const auto&... gs) {
            return ranges::views::cartesian_product(bin_volume_view(gs)...) |
                ranges::views::transform([](const auto& tup) {
                    return std::apply([](const auto&... vols) { return (1 * ... * vols); }, tup);
                });
        },
        grid.grids());
}

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_UTILS_HPP
