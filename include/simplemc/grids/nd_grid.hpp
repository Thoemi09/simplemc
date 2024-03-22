/**
 * @file
 * @brief N-dimensional grid.
 */

#ifndef SIMPLEMC_GRIDS_ND_GRID_HPP
#define SIMPLEMC_GRIDS_ND_GRID_HPP

#include <simplemc/grids/grid_base.hpp>

#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/transform.hpp>

#include <array>
#include <tuple>

namespace simplemc {

/**
 * @brief N-dimensional grid consisting of N 1-dimensional grids.
 *
 * @details The grids need not be of the same type.
 *
 * @tparam Grids Grid types.
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
     * @brief N-dimensional size type.
     */
    using nd_size_type = std::array<size_type, dim()>;

    /**
     * @brief N-dimensional value type.
     */
    using nd_value_type = std::array<value_type, dim()>;

    /**
     * @brief Constructor for an N-dimensional grid.
     *
     * @param gs 1-dimensional grids.
     */
    nd_grid(const Grids&... gs) : grids_(gs...) {}

    /**
     * @brief Get all grids.
     *
     * @return Reference to tuple of grids.
     */
    tuple_type& grids() { return grids_; }

    /// See nd_grid::grids().
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
     * @brief Get size of N-dimensional grid.
     *
     * @return Total number of grid points.
     */
    [[nodiscard]] size_type size() const {
        return std::apply([](const auto&... gs) { return (1 * ... * gs.size()); }, grids_);
    }

    /**
     * @brief Get shape of the grid.
     *
     * @return Array containing the size in each dimension.
     */
    [[nodiscard]] nd_size_type shape() const {
        return std::apply([](const auto&... gs) { return nd_size_type { gs.size()... }; }, grids_);
    }

    /**
     * @brief Get grid point at a given index array.
     *
     * @param idx_arr Index array.
     * @return Grid point at that index array.
     */
    [[nodiscard]] nd_value_type at(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->at(args...); }, idx_arr);
    }

    /**
     * @brief Get grid point at the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices for each dimension.
     * @return Grid point at those indices.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type at(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.at(idxs)... }; }, grids_);
    }

    /**
     * @brief Center of a given bin volume.
     *
     * @param idx_arr Index array.
     * @return Center of bin specified by index array.
     */
    [[nodiscard]] nd_value_type center(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->center(args...); }, idx_arr);
    }

    /**
     * @brief Center of a given bin volume.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices for each dimension.
     * @return Center of bin specified by indices.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type center(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.center(idxs)... }; }, grids_);
    }

    /**
     * @brief Get index array of the bin to which a given value array belongs.
     *
     * @param val_arr Value array.
     * @return Index array.
     */
    [[nodiscard]] nd_size_type index(const nd_value_type& val_arr) const {
        return std::apply([this](const auto&... args) { return this->index(args...); }, val_arr);
    }

    /**
     * @brief Get index array of bin to which the given values belong.
     *
     * @tparam Vals Value types.
     * @param vals Values for each dimension.
     * @return Index array.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index(Vals... vals) const {
        return std::apply([vals...](const auto&... gs) { return nd_size_type { gs.index(vals)... }; }, grids_);
    }

    /**
     * @brief Call grid_base::index_subrange for each grid for a given value array.
     *
     * @param m Size of subrange.
     * @param val_array Value array.
     * @return Index array.
     */
    [[nodiscard]] nd_size_type index_subrange(size_type m, const nd_value_type& val_arr) const {
        return std::apply([m, this](const auto&... args) { return this->index_subrange(m, args...); }, val_arr);
    }

    /**
     * @brief Call grid_base::index_subrange for each grid for the given values.
     *
     * @tparam Vals Value types.
     * @param m Size of subrange.
     * @param vals Values for each dimension.
     * @return Index array.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index_subrange(size_type m, Vals... vals) const {
        return std::apply(
            [m, vals...](const auto&... gs) { return nd_size_type { gs.index_subrange(m, vals)... }; }, grids_);
    }

    /**
     * @brief Get bin volume at a given index array.
     *
     * @param idx_arr Index array
     * @return Volume of bin.
     */
    [[nodiscard]] value_type bin_volume(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->bin_volume(args...); }, idx_arr);
    }

    /**
     * @brief Get bin volume at the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices for each dimension.
     * @return Volume of bin.
     */
    template <typename... Idxs>
    [[nodiscard]] value_type bin_volume(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return (1 * ... * gs.bin_volume(idxs)); }, grids_);
    }

    /**
     * @brief Get a view on the grid.
     *
     * @details The grid is traversed in row-major order, i.e. the last dimension is the fastest varying one and
     * the first dimension is the slowest varying one.
     *
     * @return View on the N-dimensional grid.
     */
    [[nodiscard]] auto view() const {
        return std::apply(
            [](const auto&... gs) {
                return ranges::views::cartesian_product(gs.view()...) | ranges::views::transform([](const auto& tup) {
                    return std::apply([](const auto&... vals) { return nd_value_type { vals... }; }, tup);
                });
            },
            grids_);
    }

    /**
     * @brief Get a view on the centered grid.
     *
     * @details The grid is traversed in row-major order, i.e. the last dimension is the fastest varying one and
     * the first dimension is the slowest varying one.
     *
     * @return View on the N-dimensional centered grid.
     */
    [[nodiscard]] auto view_center() const {
        return std::apply(
            [](const auto&... gs) {
                return ranges::views::cartesian_product(gs.view_center()...) |
                    ranges::views::transform([](const auto& tup) {
                        return std::apply([](const auto&... vals) { return nd_value_type { vals... }; }, tup);
                    });
            },
            grids_);
    }

    /**
     * @brief Get a view on the bin volumes.
     *
     * @details The grid is traversed in row-major order, i.e. the last dimension is the fastest varying one and
     * the first dimension is the slowest varying one.
     *
     * @return View on the bin volumes.
     */
    [[nodiscard]] auto view_bin_volumes() const {
        return std::apply(
            [](const auto&... gs) {
                return ranges::views::cartesian_product(gs.view_bin_volumes()...) |
                    ranges::views::transform([](const auto& tup) {
                        return std::apply([](const auto&... vols) { return (1 * ... * vols); }, tup);
                    });
            },
            grids_);
    }

private:
    tuple_type grids_;
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_ND_GRID_HPP
