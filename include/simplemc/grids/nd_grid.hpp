/**
 * @file
 * @brief Generic N-dimensional grid.
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
 * @ingroup simplemc-grids-nd
 * @brief Generic N-dimensional grid consisting of \f$ N \f$ 1-dimensional grids.
 *
 * @details The grids need not be of the same type as long as they have the same interface as
 * simplemc::grid_base.
 *
 * It stores the 1-dimensional grids in a tuple.
 *
 * For example, a 2-dimensional, 3x3 linear grid can be constructed as follows:
 *
 * @code{.cpp}
 * simplemc::linear_grid lg(0.0, 1.0, 3);
 * simplemc::nd_grid grid_2d(lg, lg);
 * fmt::print("{}\n", grid_2d.view());
 * @endcode
 *
 * Output:
 *
 * ```
 * [[0, 0], [0, 0.5], [0, 1], [0.5, 0], [0.5, 0.5], [0.5, 1], [1, 0], [1, 0.5], [1, 1]]
 * ```
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
     * @brief Type of the range of the 1-dimensional grids.
     */
    using value_type = grid_base::value_type;

    /**
     * @brief Type of the domain of the 1-dimensional grids.
     */
    using size_type = grid_base::size_type;

    /**
     * @brief Type of the N-dimensional grid's range, i.e. the type of the grid points.
     */
    using nd_size_type = std::array<size_type, dim()>;

    /**
     * @brief Type of the N-dimensional grid's domain, i.e. the type of the grid indices.
     */
    using nd_value_type = std::array<value_type, dim()>;

    /**
     * @brief Construct an N-dimensional grid by specifying its 1-dimensional grids.
     *
     * @param gs 1-dimensional grids.
     */
    nd_grid(const Grids&... gs) : grids_(gs...) {}

    /**
     * @brief Get all 1-dimensional grids.
     *
     * @return Reference to the tuple of grids.
     */
    tuple_type& grids() { return grids_; }

    /**
     * @brief Get all 1-dimensional grids.
     *
     * @return Const reference to the tuple of grids.
     */
    const tuple_type& grids() const { return grids_; }

    /**
     * @brief Get the first grid point.
     *
     * @return First value of the grid.
     */
    [[nodiscard]] nd_value_type first() const {
        return std::apply([](const auto&... gs) { return nd_value_type { gs.first()... }; }, grids_);
    }

    /**
     * @brief Get the last grid point.
     *
     * @return Last value of the grid.
     */
    [[nodiscard]] nd_value_type last() const {
        return std::apply([](const auto&... gs) { return nd_value_type { gs.last()... }; }, grids_);
    }

    /**
     * @brief Get the grid size.
     *
     * @return Number of grid points on the grid.
     */
    [[nodiscard]] size_type size() const {
        return std::apply([](const auto&... gs) { return (1 * ... * gs.size()); }, grids_);
    }

    /**
     * @brief Get the shape of the N-dimensional grid.
     *
     * @return 'std::array' containing the sizes of the 1-dimensional grids.
     */
    [[nodiscard]] nd_size_type shape() const {
        return std::apply([](const auto&... gs) { return nd_size_type { gs.size()... }; }, grids_);
    }

    /**
     * @brief Get the grid point at a given index array.
     *
     * @param idx_arr Index array specifying the grid point.
     * @return Grid point at the given index array.
     */
    [[nodiscard]] nd_value_type at(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->at(args...); }, idx_arr);
    }

    /**
     * @brief Get the grid point at the given indices.
     *
     * @details It simply calls at() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices for all dimensions.
     * @return Grid point at the given indices.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type at(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.at(idxs)... }; }, grids_);
    }

    /**
     * @brief Get the center of the bin at a given index array.
     *
     * @param idx_arr Index array specifying the bin.
     * @return Center of the bin at the given index array.
     */
    [[nodiscard]] nd_value_type center(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->center(args...); }, idx_arr);
    }

    /**
     * @brief Get the center of the bin at the given indices.
     *
     * @details It simply calls center() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices for all dimensions.
     * @return Center of the bin at the given indices.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type center(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.center(idxs)... }; }, grids_);
    }

    /**
     * @brief Get the index array of the bin to which a given N-dimensional point belongs.
     *
     * @param val_arr Value array specifying an N-dimensional point in the range of the grid.
     * @return Index array of the bin which contains the given point.
     */
    [[nodiscard]] nd_size_type index(const nd_value_type& val_arr) const {
        return std::apply([this](const auto&... args) { return this->index(args...); }, val_arr);
    }

    /**
     * @brief Get the index array of the bin to which a given N-dimensional point belongs.
     *
     * @details It simply calls index() with a value array constructed from the given values.
     *
     * @tparam Vals Value types.
     * @param vals Values for all dimensions specifying an N-dimensional point in the range of the
     * grid.
     * @return Index array of the bin which contains the given point.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index(Vals... vals) const {
        return std::apply([vals...](const auto&... gs) { return nd_size_type { gs.index(vals)... }; }, grids_);
    }

    /**
     * @brief Find the subvolume of \f$ m - 1 \f$ consecutive bins such that the given N-dimensional
     * point lies more or less in its middle.
     *
     * @details It simply calls simplemc::grid_base::index_subrange for each 1-dimensional grid.
     *
     * @param m Size of the wanted subrange.
     * @param val_arr Value array specifying an N-dimensional point in the range of the grid.
     * @return Index array of the bin of the subvolume with the smallest indices.
     */
    [[nodiscard]] nd_size_type index_subrange(size_type m, const nd_value_type& val_arr) const {
        return std::apply([m, this](const auto&... args) { return this->index_subrange(m, args...); }, val_arr);
    }

    /**
     * @brief Find the subvolume of \f$ m - 1 \f$ consecutive bins such that the given N-dimensional
     * point lies more or less in its middle.
     *
     * @details It simply calls simplemc::grid_base::index_subrange for each 1-dimensional grid.
     *
     * @tparam Vals Value types.
     * @param m Size of the wanted subrange.
     * @param vals Values for all dimensions specifying an N-dimensional point in the range of the
     * grid.
     * @return Index array of the bin of the subvolume with the smallest indices.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index_subrange(size_type m, Vals... vals) const {
        return std::apply(
            [m, vals...](const auto&... gs) { return nd_size_type { gs.index_subrange(m, vals)... }; }, grids_);
    }

    /**
     * @brief Get the volume of the bin at the given index array.
     *
     * @param idx_arr Index array specifying the bin.
     * @return Bin volume at the given index array.
     */
    [[nodiscard]] value_type bin_volume(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->bin_volume(args...); }, idx_arr);
    }

    /**
     * @brief Get the volume of the bin at the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices for all dimensions.
     * @return Bin volume at the given indices.
     */
    template <typename... Idxs>
    [[nodiscard]] value_type bin_volume(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return (1 * ... * gs.bin_volume(idxs)); }, grids_);
    }

    /**
     * @brief Get a lazy view on the grid.
     *
     * @details The grid is traversed in row-major (C) order, i.e. the last dimension is the fastest
     * varying one and the first dimension is the slowest varying one.
     *
     * @return 'ranges::view' on the N-dimensional grid points.
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
     * @brief Get a lazy view on the bin centers.
     *
     * @details The grid is traversed in row-major (C) order, i.e. the last dimension is the fastest
     * varying one and the first dimension is the slowest varying one.
     *
     * @return 'ranges::view' on the N-dimensional bin centers.
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
     * @brief Get a lazy view on the bin volumes.
     *
     * @details The grid is traversed in row-major (C) order, i.e. the last dimension is the fastest
     * varying one and the first dimension is the slowest varying one.
     *
     * @return 'ranges::view' on the N-dimensional bin volumes.
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
