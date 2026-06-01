/**
 * @file
 * @brief Generic N-dimensional grid.
 */

#ifndef SIMPLEMC_GRIDS_ND_GRID_HPP
#define SIMPLEMC_GRIDS_ND_GRID_HPP

#include <simplemc/grids/concepts.hpp>
#include <simplemc/grids/grid_iterator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <array>
#include <utility>
#include <tuple>

namespace simplemc {

/**
 * @ingroup simplemc-grids-nd
 * @brief Generic N-dimensional grid consisting of \f$ N \f$ @ref simplemc-grids-1d.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-nd.
 *
 * A simplemc::nd_grid satisfies the simplemc::grid_nd concept. The underlying 1-dimensional grids
 * need not be of the same type as long as they satisfy the simplemc::grid_1d concept. They are stored
 * in a `std::tuple` and can be accessed via the grids() member function.
 *
 * We use `std::array<long, N>` types to represent multi-dimensional indices \f$ \mathbf{i} = (i_1,
 * \dots, i_N) \in \mathrm{I} \f$ and `std::array<double, N>` types to represent grid points and
 * general points \f$ \mathbf{x} = (x_1, \dots, x_N) \in \mathrm{R} \f$.
 *
 * When iterating over views of grid points, bin centers or bin volumes, the elements are traversed in
 * row-major (C) order (see simplemc::row_major).
 *
 * @include grids/doc_nd_grid.cpp
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
template <grid_1d... Grids>
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
     * @brief Tuple type of 1-dimensional grids.
     */
    using tuple_type = std::tuple<Grids...>;

    /**
     * @brief Type of the grid's range \f$ \mathrm{R} \f$, i.e. the type of the grid points.
     */
    using value_type = std::array<double, dim()>;

    /**
     * @brief Type of the grid's domain \f$ \mathrm{I} \f$, i.e. the type of the grid indices.
     */
    using size_type = std::array<long, dim()>;

    /**
     * @brief Default constructor calls the default constructor of each 1-dimensional grid.
     */
    constexpr nd_grid() = default;

    /**
     * @brief Construct an N-dimensional grid by specifying its 1-dimensional grids.
     *
     * @param gs simplemc::grid_1d grids.
     */
    constexpr nd_grid(Grids... gs) : grids_(std::move(gs)...) {}

    /**
     * @brief Get all 1-dimensional grids.
     *
     * @return Reference to the tuple of grids.
     */
    constexpr tuple_type& grids() noexcept { return grids_; }

    /**
     * @brief Get all 1-dimensional grids.
     *
     * @return Const reference to the tuple of grids.
     */
    constexpr const tuple_type& grids() const noexcept { return grids_; }

    /**
     * @brief Get the first grid point \f$ \mathbf{a} = (a_1, \dots, a_N) \f$.
     *
     * @return Value of \f$ g(0, \dots, 0) = (g_1(0), \dots, g_N(0)) \f$.
     */
    [[nodiscard]] constexpr value_type first() const {
        return std::apply([](const auto&... gs) { return value_type { gs.at(0)... }; }, grids_);
    }

    /**
     * @brief Get the last grid point \f$ \mathbf{b} = (b_1, \dots, b_N) \f$.
     *
     * @return Value of \f$ g(M_1 - 1, \dots, M_N - 1) = (g_1(M_1 - 1), \dots, g_N(M_N - 1)) \f$.
     */
    [[nodiscard]] constexpr value_type last() const {
        return std::apply([](const auto&... gs) { return value_type { gs.at(gs.size() - 1)... }; }, grids_);
    }

    /**
     * @brief Get the grid size.
     *
     * @return Number of grid points \f$ M = M_1 \times \dots \times M_N \f$ on the grid.
     */
    [[nodiscard]] constexpr long size() const {
        return std::apply([](const auto&... gs) { return (1 * ... * gs.size()); }, grids_);
    }

    /**
     * @brief Get the shape of the N-dimensional grid.
     *
     * @return `std::array` containing the sizes of all 1-dimensional grids, i.e. \f$ (M_1, \dots,
     * M_N) \f$.
     */
    [[nodiscard]] constexpr size_type shape() const {
        return std::apply([](const auto&... gs) { return size_type { gs.size()... }; }, grids_);
    }

    /**
     * @brief Get the grid point \f$ g(\mathbf{i}) \f$ at a given index array \f$ \mathbf{i} =
     * (i_1, \dots, i_N) \f$.
     *
     * @param idx_arr Index array \f$ \mathbf{i} \in \mathrm{I} \f$ specifying the grid point.
     * @return Grid point \f$ g(\mathbf{i}) = (g_1(i_1), \dots, g_N(i_N)) \f$.
     */
    [[nodiscard]] constexpr value_type at(const size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->at(args...); }, idx_arr);
    }

    /**
     * @brief Get the grid point \f$ g(\mathbf{i}) \f$ at a given index array \f$ \mathbf{i} =
     * (i_1, \dots, i_N) \f$.
     *
     * @details It simply calls at() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices \f$ i_1, \dots, i_N \f$.
     * @return Grid point \f$ g(\mathbf{i}) = (g_1(i_1), \dots, g_N(i_N)) \f$.
     */
    template <typename... Idxs>
    [[nodiscard]] constexpr value_type at(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return value_type { gs.at(idxs)... }; }, grids_);
    }

    /**
     * @brief Get the index array \f$ \mathbf{i} = (i_1, \dots, i_N) \f$ such that a given
     * N-dimensional point \f$ \mathbf{x} \in \mathrm{R} \f$ lies in the bin \f$ b_{i_1, \dots, i_N}
     * \f$.
     *
     * @param x N-dimensional point \f$ \mathbf{x} = (x_1, \dots, x_N) \f$.
     * @return Index array \f$ \mathbf{i} = \tilde{g}^{-1}(x_1, \dots, x_N) \f$ of the bin which
     * contains the given point.
     */
    [[nodiscard]] constexpr size_type index(const value_type& x) const {
        return std::apply([this](const auto&... args) { return this->index(args...); }, x);
    }

    /**
     * @brief Get the index array \f$ \mathbf{i} = (i_1, \dots, i_N) \f$ such that a given
     * N-dimensional point \f$ \mathbf{x} \in \mathrm{R} \f$ lies in the bin \f$ b_{i_1, \dots, i_N}
     * \f$.
     *
     * @details It simply calls index() with a value array constructed from the given values.
     *
     * @tparam Vals Value types.
     * @param xs 1-dimensional points \f$ x_1, \dots, x_N \f$.
     * @return Index array \f$ \mathbf{i} = \tilde{g}^{-1}(x_1, \dots, x_N) \f$ of the bin which
     * contains the given point.
     */
    template <typename... Vals>
    [[nodiscard]] constexpr size_type index(Vals... xs) const {
        return std::apply([xs...](const auto&... gs) { return size_type { gs.index(xs)... }; }, grids_);
    }

    /**
     * @brief Get the volume of the bin at the given index array \f$ \mathbf{i} = (i_1, \dots, i_N)
     * \f$.
     *
     * @details The volume of the N-dimensional bin is the product of the volumes of the 1-dimensional
     * bins, i.e. \f$ \mathrm{Vol}(b_{i_1, \dots, i_N}) = \mathrm{Vol}(b_{i_1}) \times \dots \times
     * \mathrm{Vol}(b_{i_N}) \f$.
     *
     * @param idx_arr Index array \f$ \mathbf{i} \in \mathrm{I} \f$ specifying the bin.
     * @return Volume (size) of the bin \f$ b_{i_1, \dots, i_N} \f$.
     */
    [[nodiscard]] constexpr double volume(const size_type& idx_arr) const {
        return std::apply([this](const auto&... args) { return this->volume(args...); }, idx_arr);
    }

    /**
     * @brief Get the volume of the bin at the given indices \f$ i_1, \dots, i_N \f$.
     *
     * @details It simply calls volume() with an index array constructed from the given indices.
     *
     * @tparam Idxs Index types.
     * @param idxs Indices \f$ i_1, \dots, i_N \f$.
     * @return Volume (size) of the bin \f$ b_{i_1, \dots, i_N} \f$.
     */
    template <typename... Idxs>
    [[nodiscard]] constexpr double volume(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return (1 * ... * gs.volume(idxs)); }, grids_);
    }

    /**
     * @brief Get the center of the bin at a given index array \f$ \mathbf{i} = (i_1, \dots, i_N) \f$.
     *
     * @details The center \f$ \mathbf{c} \f$ of the N-dimensional bin \f$ b_{i_1, \dots, i_N} \f$ is
     * \f$ \mathbf{c} = (c_{i_1}, \dots, c_{i_N}) \f$, where \f$ c_{i_k} \f$ is the center of the bin
     * \f$ b_{i_k} \f$ of the 1-dimensional grid \f$ g_k \f$.
     *
     * @param idx_arr Index array \f$ \mathbf{i} \in \mathrm{I} \f$ specifying the bin.
     * @return Center of the bin \f$ b_{i_1, \dots, i_N} \f$.
     */
    [[nodiscard]] constexpr value_type center(const size_type& idx_arr) const {
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
    [[nodiscard]] constexpr value_type center(Idxs... idxs) const {
        return std::apply([idxs...](const auto&... gs) { return value_type { gs.center(idxs)... }; }, grids_);
    }

    /**
     * @brief Get an iterator to the first grid point.
     *
     * @return Iterator to the first grid point \f$ g(0, \dots, 0) \f$.
     */
    [[nodiscard]] constexpr auto begin() const noexcept { return grid_iterator<nd_grid> { *this }; }

    /**
     * @brief Get a const iterator to the first grid point.
     *
     * @return Const iterator to the first grid point \f$ g(0, \dots, 0) \f$.
     */
    [[nodiscard]] constexpr auto cbegin() const noexcept { return begin(); }

    /**
     * @brief Get an iterator to one past the last grid point.
     *
     * @return Iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto end() const noexcept { return grid_iterator<nd_grid> { *this, size() }; }

    /**
     * @brief Get a const iterator to one past the last grid point.
     *
     * @return Const iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    tuple_type grids_;
};

/// `nd_grid` is just a tuple of 1-D grids; @ref save_tuple / @ref load_tuple do the work.
template <class S, grid_1d... Grids>
    requires serializer<std::remove_cvref_t<S>>
void simplemc_save(S&& s, const nd_grid<Grids...>& g) {
    save_tuple(s["grids"], g.grids());
}

template <class S, grid_1d... Grids>
    requires deserializer<std::remove_cvref_t<S>>
void simplemc_load(S&& s, nd_grid<Grids...>& g) {
    load_tuple(s["grids"], g.grids());
}

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_ND_GRID_HPP
