/**
 * @file
 * @brief Generic N-dimensional grid.
 */

#ifndef SIMPLEMC_GRIDS_ND_GRID_HPP
#define SIMPLEMC_GRIDS_ND_GRID_HPP

#include <simplemc/grids/grid_base.hpp>
#include <simplemc/utils/ranges.hpp>

#include <array>
#include <utility>
#include <tuple>

namespace simplemc {

/**
 * @ingroup simplemc-grids-nd
 * @brief Generic N-dimensional grid consisting of \f$ N \f$ @ref simplemc-grids-1d.
 *
 * @details The 1-dimensional grids need not be of the same type as long as they have the same
 * interface as simplemc::grid_base. They are stored in a `std::tuple` and can be accessed via the
 * grids() member function.
 *
 * N-dimensional indices \f$ i = (i_1, \dots, i_N) \in \mathrm{I} \f$ are represented as
 * `std::array<size_type, N>` objects. Similarly, grid points and general points \f$ x = (x_1, \dots,
 * x_N) \in \mathrm{R} \f$ are represented as `std::array<value_type, N>` objects.
 *
 * When iterating over views of grid points, bin centers or bin volumes, the elements are traversed in
 * row-major (C) order (see simplemc::row_major).
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/grids.hpp>
 *
 * int main() {
 *     // create a linear grid of size 3 on [0, 1]
 *     simplemc::linear_grid lg { 0.0, 1.0, 3 };
 *
 *     // create a quadratic grid of size 3 on [0, 0.5]
 *     simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
 *
 *     // combine the two 1d grids into a 2d grid
 *     simplemc::nd_grid grid_2d { lg, pg };
 *
 *     // print some basic information
 *     fmt::println("Size: {}, Shape: {}", grid_2d.size(), grid_2d.shape());
 *
 *     // print the grid points, bin centers and bin volumes
 *     fmt::println("Grid points: {}", grid_2d.view());
 *     fmt::println("Bin centers: {}", grid_2d.view_center());
 *     fmt::println("Bin volumes: {}", grid_2d.view_bin_volumes());
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Size: 9, Shape: [3, 3]
 * Grid points: [[0, 0], [0, 0.125], [0, 0.5], [0.5, 0], [0.5, 0.125], [0.5, 0.5], [1, 0], [1, 0.125], [1, 0.5]]
 * Bin centers: [[0.25, 0.0625], [0.25, 0.3125], [0.75, 0.0625], [0.75, 0.3125]]
 * Bin volumes: [0.0625, 0.1875, 0.0625, 0.1875]
 * ```
 *
 * @tparam Grids Grid types.
 */
template <typename... Grids>
    requires(sizeof...(Grids) > 0)
class nd_grid {
public:
    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions (\f$ = N\f$).
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
     * @brief Default constructor calls the default constructor of each 1-dimensional grid.
     */
    nd_grid() = default;

    /**
     * @brief Construct an N-dimensional grid by specifying its 1-dimensional grids.
     *
     * @param gs 1-dimensional grids.
     */
    nd_grid(Grids... gs) : grids_(std::move(gs)...) {}

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
     * @brief Get the first grid point \f$ a = (a_1, \dots, a_N) \f$.
     *
     * @return Value of \f$ g(0, \dots, 0) = (g_1(0), \dots, g_N(0)) \f$.
     */
    [[nodiscard]] nd_value_type first() const {
        return std::apply([](const auto&... gs) { return nd_value_type { gs.first()... }; }, grids_);
    }

    /**
     * @brief Get the last grid point \f$ b = (b_1, \dots, b_N) \f$.
     *
     * @return Value of \f$ g(M_1 - 1, \dots, M_N - 1) = (g_1(M_1 - 1), \dots, g_N(M_N - 1)) \f$.
     */
    [[nodiscard]] nd_value_type last() const {
        return std::apply([](const auto&... gs) { return nd_value_type { gs.last()... }; }, grids_);
    }

    /**
     * @brief Get the grid size.
     *
     * @return Number of grid points \f$ M = M_1 \times \dots \times M_N \f$ on the grid.
     */
    [[nodiscard]] size_type size() const {
        return std::apply([](const auto&... gs) { return (1 * ... * gs.size()); }, grids_);
    }

    /**
     * @brief Get the shape of the N-dimensional grid.
     *
     * @return `std::array` containing the sizes of all 1-dimensional grids, i.e. \f$ (M_1, \dots,
     * M_N) \f$.
     */
    [[nodiscard]] nd_size_type shape() const {
        return std::apply([](const auto&... gs) { return nd_size_type { gs.size()... }; }, grids_);
    }

    /**
     * @brief Get the grid point \f$ g(i_1, \dots, i_N) \f$ at a given index array \f$ i = (i_1,
     * \dots, i_N) \f$.
     *
     * @param idx_arr Index array \f$ i = (i_1, \dots, i_N) \f$ specifying the grid point.
     * @return Grid point \f$ g(i_1, \dots, i_N) = (g_1(i_1), \dots, g_N(i_N)) \f$.
     */
    [[nodiscard]] nd_value_type at(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->at(args...); }, idx_arr);
    }

    /**
     * @brief Get the grid point \f$ g(i_1, \dots, i_N) \f$ at the given indices \f$ i_1, \dots, i_N
     * \f$.
     *
     * @details It simply calls at() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices \f$ i_1, \dots, i_N \f$.
     * @return Grid point \f$ g(i_1, \dots, i_N) = (g_1(i_1), \dots, g_N(i_N))\f$.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type at(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.at(idxs)... }; }, grids_);
    }

    /**
     * @brief Get the center of the bin at a given index array \f$ i = (i_1, \dots, i_N) \f$.
     *
     * @details The center \f$ c \f$ of the N-dimensional bin \f$ b_{i_1, \dots, i_N} \f$ is \f$ c =
     * (c_{i_1}, \dots, c_{i_N}) \f$, where \f$ c_{i_k} \f$ is the center of the bin \f$ b_{i_k} \f$
     * of the 1-dimensional grid \f$ g_k \f$.
     *
     * @param idx_arr Index array \f$ i = (i_1, \dots, i_N) \f$ specifying the bin.
     * @return Center of the bin \f$ b_{i_1, \dots, i_N} \f$.
     */
    [[nodiscard]] nd_value_type center(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->center(args...); }, idx_arr);
    }

    /**
     * @brief Get the center of the bin at the given indices \f$ i_1, \dots, i_N \f$.
     *
     * @details It simply calls center() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices \f$ i_1, \dots, i_N \f$.
     * @return Center of the bin \f$ b_{i_1, \dots, i_N} \f$.
     */
    template <typename... Idxs>
    [[nodiscard]] nd_value_type center(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return nd_value_type { gs.center(idxs)... }; }, grids_);
    }

    /**
     * @brief Get the index array \f$ i = (i_1, \dots, i_N) \f$ such that a given N-dimensional point
     * \f$ x \in \mathrm{R} \f$ lies in the bin \f$ b_{i_1, \dots, i_N} \f$.
     *
     * @param val_arr Value array \f$ x = (x_1, \dots, x_N) \f$.
     * @return Index array \f$ i = \tilde{g}^{-1}(x_1, \dots, x_N) \f$ of the bin which contains the
     * given point.
     */
    [[nodiscard]] nd_size_type index(const nd_value_type& val_arr) const {
        return std::apply([this](const auto&... args) { return this->index(args...); }, val_arr);
    }

    /**
     * @brief Get the index array \f$ i = (i_1, \dots, i_N) \f$ such that a given N-dimensional point
     * \f$ x \in \mathrm{R} \f$ lies in the bin \f$ b_{i_1, \dots, i_N} \f$.
     *
     * @details It simply calls index() with a value array constructed from the given values.
     *
     * @tparam Vals Value types.
     * @param vals Values \f$ x_1, \dots, x_N \f$.
     * @return Index array \f$ i = \tilde{g}^{-1}(x_1, \dots, x_N) \f$ of the bin which contains the
     * given point.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index(Vals... vals) const {
        return std::apply([vals...](const auto&... gs) { return nd_size_type { gs.index(vals)... }; }, grids_);
    }

    /**
     * @brief Find the subvolume of \f$ m - 1 \f$ consecutive bins such that the given N-dimensional
     * point \f$ x \in \mathrm{R} \f$ lies more or less in its middle.
     *
     * @details It calls simplemc::grid_base::index_subrange for each 1-dimensional grid.
     *
     * @param m Size of the wanted subrange.
     * @param val_arr Value array \f$ x = (x_1, \dots, x_N) \f$.
     * @return Index array \f$ i = (i_1, \dots, i_N) \f$ of the bin of the subvolume with the smallest
     * indices.
     */
    [[nodiscard]] nd_size_type index_subrange(size_type m, const nd_value_type& val_arr) const {
        return std::apply([m, this](const auto&... args) { return this->index_subrange(m, args...); }, val_arr);
    }

    /**
     * @brief Find the subvolume of \f$ m - 1 \f$ consecutive bins such that the given N-dimensional
     * point lies more or less in its middle.
     *
     * @details It simply calls index_subrange() with a value array constructed from the given values.
     *
     * @tparam Vals Value types.
     * @param m Size of the wanted subrange.
     * @param vals Values \f$ x_1, \dots, x_N \f$.
     * @return Index array \f$ i = (i_1, \dots, i_N) \f$ of the bin of the subvolume with the smallest
     * indices.
     */
    template <typename... Vals>
    [[nodiscard]] nd_size_type index_subrange(size_type m, Vals... vals) const {
        return std::apply(
            [m, vals...](const auto&... gs) { return nd_size_type { gs.index_subrange(m, vals)... }; }, grids_);
    }

    /**
     * @brief Get the volume of the bin at the given index array \f$ i = (i_1, \dots, i_N) \f$.
     *
     * @details The volume of the N-dimensional bin is the product of the volumes of the 1-dimensional
     * bins, i.e. \f$ \mathrm{Vol}(b_{i_1, \dots, i_N}) = \mathrm{Vol}(b_{i_1}) \times \dots \times
     * \mathrm{Vol}(b_{i_N}) \f$.
     *
     * @param idx_arr Index array \f$ i = (i_1, \dots, i_N) \f$ specifying the bin.
     * @return Volume (size) of the bin \f$ b_{i_1, \dots, i_N} \f$.
     */
    [[nodiscard]] value_type bin_volume(const nd_size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->bin_volume(args...); }, idx_arr);
    }

    /**
     * @brief Get the volume of the bin at the given indices \f$ i_1, \dots, i_N \f$.
     *
     * @details It simply calls bin_volume() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices  \f$ i_1, \dots, i_N \f$.
     * @return Volume (size) of the bin \f$ b_{i_1, \dots, i_N} \f$.
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
     * @code{.cpp}
     * // create a 2d grid
     * simplemc::linear_grid lg { 0.0, 1.0, 3 };
     * simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
     * simplemc::nd_grid grid_2d { lg, pg };
     *
     * // print the grid points as a view
     * fmt::println("{}", grid_2d.view());
     * @endcode
     *
     * Output:
     *
     * ```
     * [[0, 0], [0, 0.125], [0, 0.5], [0.5, 0], [0.5, 0.125], [0.5, 0.5], [1, 0], [1, 0.125], [1, 0.5]]
     * ```
     *
     * @return View on the N-dimensional grid points.
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
     * @code{.cpp}
     * // create a 2d grid
     * simplemc::linear_grid lg { 0.0, 1.0, 3 };
     * simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
     * simplemc::nd_grid grid_2d { lg, pg };
     *
     * // print the bin centers as a view
     * fmt::println("{}", grid_2d.view_center());
     * @endcode
     *
     * Output:
     *
     * ```
     * [[0.25, 0.0625], [0.25, 0.3125], [0.75, 0.0625], [0.75, 0.3125]]
     * ```
     *
     * @return View on the N-dimensional bin centers.
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
     * @code{.cpp}
     * // create a 2d grid
     * simplemc::linear_grid lg { 0.0, 1.0, 3 };
     * simplemc::power_grid pg { 0.0, 0.5, 3, 2 };
     * simplemc::nd_grid grid_2d { lg, pg };
     *
     * // print the bin volumes as a view
     * fmt::println("{}", grid_2d.view_bin_volumes());
     * @endcode
     *
     * Output:
     *
     * ```
     * [0.0625, 0.1875, 0.0625, 0.1875]
     * ```
     *
     * @return View on the N-dimensional bin volumes.
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
