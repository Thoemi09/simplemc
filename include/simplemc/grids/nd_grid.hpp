/**
 * @file nd_grid.hpp
 * @brief N-dimensional grid.
 */

#ifndef SIMPLEMC_GRIDS_ND_GRID_HPP
#define SIMPLEMC_GRIDS_ND_GRID_HPP

#include <simplemc/grids/grid_base.hpp>

#include <range/v3/view/cartesian_product.hpp>

#include <array>
#include <tuple>

namespace simplemc {

/**
 * @brief N-dimensional grid consisting of N 1-dimensional grids.
 *
 * @tparam Grids N grid types.
 */
template <typename... Grids>
class nd_grid {
    static_assert(sizeof...(Grids) > 0, "0 dimensions in nd_grid not allowed");

public:
    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions.
     */
    static constexpr std::size_t dim() { return std::tuple_size_v<tuple_type>; }

    /**
     * @brief Tuple of grids.
     */
    using tuple_type = std::tuple<Grids...>;

    /**
     * @brief Value type of each 1-dimensional grid.
     */
    using value_type = grid_base::value_type;

    /**
     * @brief Size type of each 1-dimensional grid.
     */
    using size_type = grid_base::size_type;

    /**
     * @brief Array of indices/sizes.
     */
    using nd_size_type = std::array<size_type, dim()>;

    /**
     * @brief N-dimensional value type.
     */
    using nd_value_type = std::array<value_type, dim()>;

    /**
     * @brief Constructor for an nd-grid.
     *
     * @param gs N grids.
     */
    nd_grid(const Grids&... gs) : grids_(gs...) {}

    /**
     * @brief Get all grids.
     *
     * @return Refernce to tuple of grids.
     */
    tuple_type& grids() { return grids_; }

    const tuple_type& grids() const { return grids_; }

    /**
     * @brief Get first grid point.
     *
     * @return First value of grid.
     */
    [[nodiscard]] nd_value_type first() const {
        return std::apply([](const auto&... gs) { return nd_value_type { gs.first()... }; }, grids_);
    }

    /**
     * @brief Get last grid point.
     *
     * @return Last value of grid.
     */
    [[nodiscard]] nd_value_type last() const {
        return std::apply([](const auto&... gs) { return nd_value_type { gs.last()... }; }, grids_);
    }

    /**
     * @brief Get size of nd-grid.
     *
     * @return Total number of gridpoints.
     */
    [[nodiscard]] size_type size() const {
        return std::apply([](const auto&... gs) { return (1 * ... * gs.size()); }, grids_);
    }

    /**
     * @brief Get grid point at a certain index array.
     *
     * @param idx_arr Index array.
     * @return Value array at that index array.
     */
    [[nodiscard]] nd_value_type at(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->at(args...); }, idx_arr);
    }

    /**
     * @brief Get grid point at certain indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices.
     * @return Value array at those indices.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type at(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.at(idxs)... }; }, grids_);
    }

    /**
     * @brief Center of a certain bin volume.
     *
     * @param idx_arr Index array.
     * @return Center of bin specified by index array.
     */
    [[nodiscard]] nd_value_type center(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->center(args...); }, idx_arr);
    }

    /**
     * @brief Center of a certain bin volume.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices.
     * @return Center of bin specified by indices.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type center(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.center(idxs)... }; }, grids_);
    }

    /**
     * @brief Get index array of the bin to which the value array belongs.
     *
     * @param val_arr Value array.
     * @return Index array.
     */
    [[nodiscard]] nd_size_type index(const nd_value_type& val_arr) const {
        return std::apply([this](const auto&... args) { return this->index(args...); }, val_arr);
    }

    /**
     * @brief Get index array of bin to which certain values belong.
     *
     * @tparam Vals Value types.
     * @param vals Values.
     * @return Index array.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index(Vals... vals) const {
        return std::apply([vals...](const auto&... gs) { return nd_size_type { gs.index(vals)... }; }, grids_);
    }

    /**
     * @brief Get bin volume at a certain index array.
     *
     * @param idx_arr Index array
     * @return Volume of bin.
     */
    [[nodiscard]] value_type bin_volume(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->bin_volume(args...); }, idx_arr);
    }

    /**
     * @brief Get bin size at a certain indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices.
     * @return Volume of bin.
     */
    template <typename... Idxs>
    [[nodiscard]] value_type bin_volume(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return (1 * ... * gs.bin_volume(idxs)); }, grids_);
    }

    /**
     * @brief Get a view on the grid.
     *
     * @return View on the grid.
     */
    [[nodiscard]] auto view() const {
        return std::apply([](const auto&... gs) { return ranges::views::cartesian_product(gs.view()...); }, grids_);
    }

    /**
     * @brief Get a view on the centered grid.
     *
     * @return View on the centered grid.
     */
    [[nodiscard]] auto view_center() const {
        return std::apply(
            [](const auto&... gs) { return ranges::views::cartesian_product(gs.view_center()...); }, grids_);
    }

    /**
     * @brief Get a view on the bin volumes.
     *
     * @return View on the bin volumes.
     */
    [[nodiscard]] auto view_bin_volumes() const {
        return std::apply(
            [](const auto&... gs) { return ranges::views::cartesian_product(gs.view_bin_volumes()...); }, grids_);
    }

private:
    tuple_type grids_;
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_ND_GRID_HPP
