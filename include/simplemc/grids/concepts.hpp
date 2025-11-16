/**
 * @file
 * @brief Concepts that define the requirements for grid types.
 */

#ifndef SIMPLEMC_GRIDS_CONCEPTS_HPP
#define SIMPLEMC_GRIDS_CONCEPTS_HPP

#include <simplemc/utils/ranges.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-grids
 * @brief Concept that specifies the common requirements for 1-dimensional and N-dimensional grids.
 *
 * @details We can directly map the requirements to the notation used in @ref simplemc-grids-1d and
 * @ref simplemc-grids-nd. Let \f$ i \f$ denote either a 1- or N-dimensional index from the domain \f$
 * \mathrm{I} \f$ and let \f$ x \f$ denote either a 1- or N-dimensional value from the range \f$
 * \mathrm{R} \f$. Then we have
 *
 * - `G::value_type` is the type of the range \f$ \mathrm{R} \f$
 * - `G::size_type` is the type of the domain \f$ \mathrm{I} \f$
 * - `G::dim()` returns the number of dimensions \f$ N \f$
 * - `g.size()` returns the size \f$ M \f$ of the grid
 * - `g.at(size_type)` returns the grid point \f$ g(i) \f$ at the given index \f$ i \f$
 * - `g.index(value_type)` returns the index \f$ i \f$ such that the given value \f$ x \f$ lies in the
 * bin \f$ b_i \f$
 * - `g.volume(size_type)` returns the size/volume of bin \f$ b_i \f$ with the given index \f$ i \f$
 * - `g.center(size_type)` returns the center of bin \f$ b_i \f$ with the given index \f$ i \f$
 *
 * @tparam G Type to be checked.
 */
template <typename G>
concept grid_common = requires(const G& g) {
    typename G::value_type;
    typename G::size_type;
    std::integral_constant<std::size_t, G::dim()> {};
    { g.size() } -> std::convertible_to<long>;
    { g.at(typename G::size_type {}) } -> std::same_as<typename G::value_type>;
    { g.index(typename G::size_type {}) } -> std::same_as<typename G::size_type>;
    { g.volume(typename G::size_type {}) } -> std::convertible_to<double>;
    { g.center(typename G::size_type {}) } -> std::same_as<typename G::value_type>;
};

/**
 * @ingroup simplemc-grids-1d
 * @brief Concept that specifies the requirements for @ref simplemc-grids-1d.
 *
 * @details A 1-dimensional grid satisfies the simplemc::grid_common concept and in addition has the
 * following requirements:
 *
 * - `G::value_type` is of type `double`
 * - `G::size_type` is of type `long`
 *
 * @tparam G Type to be checked.
 */
template <typename G>
concept grid_1d = grid_common<G> && std::ranges::random_access_range<G> && requires(const G& g) {
    // remove this check in case we want to relax the restrictions on value and size types
    requires std::same_as<typename G::value_type, double>;
    requires std::same_as<typename G::size_type, long>;
};

namespace detail {

// Check if a tuple has only grid_1d element types.
template <typename T>
concept has_grid_1d_element_types = []<std::size_t... Is>(std::index_sequence<Is...>) {
    return (grid_1d<std::tuple_element_t<Is, T>> && ...);
}(std::make_index_sequence<std::tuple_size_v<T>>());

} // namespace detail

/**
 * @ingroup simplemc-grids-nd
 * @brief Concept that specifies the requirements for @ref simplemc-grids-nd.
 *
 * @details An N-dimensional grid satisfies the simplemc::grid_common concept and in addition has the
 * following requirements:
 *
 * - `G::value_type` is of type `std::array<double, N>`
 * - `G::size_type` is of type `std::array<long, N>`
 * - `G::tuple_type` is a `std::tuple` of simplemc::grid_1d types
 * - `g.grids()` returns the tuple containing the underlying simplemc::grid_1d objects
 * - `g.shape()` returns the shape \f$ (M_1, \dots, M_N) \f$ of the grid
 *
 * @tparam G Type to be checked.
 */
template <typename G>
concept grid_nd = grid_common<G> && std::ranges::random_access_range<G> && requires(const G& g) {
    // remove the same_as check in case we want to relax the restrictions on value and size types
    requires std::same_as<typename G::value_type, std::array<double, G::dim()>>;
    requires std::same_as<typename G::size_type, std::array<long, G::dim()>>;
    requires detail::has_grid_1d_element_types<typename G::tuple_type>;
    { g.grids() } -> std::convertible_to<typename G::tuple_type>;
    { g.shape() } -> std::same_as<std::array<long, G::dim()>>;
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_CONCEPTS_HPP
