/**
 * @file
 * @brief Linear interpolation in arbitrary dimensions.
 */

#ifndef SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP

#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/indexing.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cstddef>
#include <span>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-interpolation
 * @{
 */

namespace detail {

// Linear interpolation given the ratio xd = (x - x0) / (x1 - x0) and the function values f0 = f(x0)
// and f1 = f(x1).
inline double interp_linear_1d(double xd, double f0, double f1) {
    return f0 * (1 - xd) + f1 * xd;
}

// Linear interpolation given the index array of the lower left corner of the hypercube, the array of
// ratios of the distances in each direction (see xd in interp_linera_1d and distance_ratios below),
// the span containing the function values, the shape of the grid and the memory order of the
// multi-dimensional array containing the function function values.
//
// This function is called recursively, with N being the current dimension.
template <int N, typename Grid, nd_order Order = column_major>
inline double interp_linear_nd(const typename Grid::nd_size_type& idx_arr, const typename Grid::nd_value_type& xd_arr,
    const std::span<double>& fvals, const typename Grid::nd_size_type& shape_arr,
    [[maybe_unused]] Order order = Order {}) {
    static_assert(N >= 0 && N < Grid::dim(), "Invalid template parameter in interp_linear_nd.");
    auto idx_p1_arr = idx_arr;
    idx_p1_arr[N] += 1;
    if constexpr (N == 0) {
        // perform simple linear interpolation in the first dimension
        const auto f0 = fvals[flat_index(idx_arr, shape_arr, order)];
        const auto f1 = fvals[flat_index(idx_p1_arr, shape_arr, order)];
        return interp_linear_1d(xd_arr[N], f0, f1);
    } else {
        // perform linear interpolation in n-1 dimensions
        const auto f0 = interp_linear_nd<N - 1, Grid>(idx_arr, xd_arr, fvals, shape_arr, order);
        const auto f1 = interp_linear_nd<N - 1, Grid>(idx_p1_arr, xd_arr, fvals, shape_arr, order);
        return interp_linear_1d(xd_arr[N], f0, f1);
    }
}

// Get ratios of the distances in each direction, i.e. (x[i] - x0[i]) / (x1[i] - x0[i]), given the
// index array of the lower left corner of the hypercube, the value array containing the point x and
// the grid.
template <typename Grid, std::size_t... Is>
inline auto distance_ratios(const typename Grid::nd_size_type& idx_arr, const typename Grid::nd_value_type& x_arr,
    const Grid& grid, std::index_sequence<Is...>) {
    return typename Grid::nd_value_type { (x_arr[Is] - std::get<Is>(grid.grids()).at(idx_arr[Is])) /
        (std::get<Is>(grid.grids()).at(idx_arr[Is] + 1) - std::get<Is>(grid.grids()).at(idx_arr[Is]))... };
}

} // namespace detail

/**
 * @brief Linear interpolation in 1D.
 *
 * @details The class stores a 1-dimensional grid (see @ref simplemc-grids-1d) together with the
 * function values at the grid points.
 *
 * @note The function values are not owned by the interpolation class. Only a span to the original
 * data is stored.
 *
 * Here is a simple example of how to create an interpolation object and use it:
 * @code
 * // exact function we want to interpolate
 * auto f = [](double x) { return x * x; };
 *
 * // define the grid on the x-axis and get the function values
 * auto grid = simplemc::linear_grid(0, 2, 50);
 * std::vector<double> fvals(50);
 * for (auto&& [x, fval] : ranges::views::zip(grid.view(), fvals)) {
 *     fval = f(x);
 * }
 *
 * // create the interpolation object
 * auto interp = simplemc::linear_interpolation(grid, fvals)
 *
 * // use the interpolation object ...
 * @endcode
 *
 * @tparam Grid 1-dimensional grid type.
 */
template <typename Grid>
class linear_interpolation {
public:
    /**
     * @brief 1-dimensional grid type.
     */
    using grid_type = Grid;

    /**
     * @brief Get the number of dimensions.
     *
     * @return Number of dimensions.
     */
    static constexpr std::size_t dim() { return 1; }

    /**
     * @brief Construct a linear interpolation object on a given grid with the given function values.
     *
     * @param grid 1-dimensional grid on the \f$ x \f$ axis.
     * @param fvals Function values at the grid points.
     */
    linear_interpolation(const grid_type& grid, const std::span<double>& fvals) : grid_(grid), fvals_(fvals) {
        if (grid.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception(
                "Number of grid points not equal to number of y values.", "linear_interpolation::linear_interplation");
        }
    }

    /**
     * @brief Perform the linear interpolation.
     *
     * @param x Value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(double x) const {
        const auto idx = grid_.index_subrange(2, x);
        const auto xd = (x - grid_.at(idx)) / (grid_.at(idx + 1) - grid_.at(idx));
        return detail::interp_linear_1d(xd, fvals_[idx], fvals_[idx + 1]);
    }

    /**
     * @brief Get the grid on the \f$ x \f$ axis.
     *
     * @return 1-dimensional grid object.
     */
    [[nodiscard]] const auto& grid() const { return grid_; }

    /**
     * @brief Get the function values.
     *
     * @return `std::span` containing the function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    grid_type grid_;
    std::span<double> fvals_;
};

/**
 * @brief Linear interpolation in \f$ N \f$ dimensions.
 *
 * @details The class stores an N-dimensional grid (see @ref simplemc-grids-nd) together with the
 * function values at the grid points.
 *
 * The second template parameter describes the memory layout of the multi-dimensional array containing
 * the function values (see simplemc::row_major and simplemc::column_major). By default, column-major
 * order is assumed.
 *
 * @note The function values are not owned by the interpolation class. Only a span to the original
 * data is stored.
 *
 * Here is a simple example of how to create an interpolation object and use it:
 * @code
 * // exact function we want to interpolate
 * auto f = [](double x, double y) { return x * x + y * y; };
 *
 * // define the 2D grid and get the function values
 * auto grid_1d = simplemc::linear_grid(0, 1, 11);
 * auto grid_2d = simplemc::nd_grid(grid_1d, grid_1d);
 * std::vector<double> fvals(11);
 * for (auto&& [arr, fval] : ranges::views::zip(grid_2d.view(), fvals)) {
 *     auto [x, y] = arr;
 *     fval = f(x, y);
 * }
 *
 * // create the interpolation object
 * auto interp = simplemc::linear_interpolation_nd(grid_2d, fvals, simplemc::row_major {});
 *
 * // use the interpolation object ...
 * @endcode
 *
 * @tparam Grid Type of N-dimensional grid.
 * @tparam Order Storage order of the multi-dimensional array containing the function values.
 */
template <typename Grid, nd_order Order = column_major>
class linear_interpolation_nd {
public:
    /**
     * @brief N-dimensional grid type.
     */
    using grid_type = Grid;

    /**
     * @brief Get the number of dimensions.
     *
     * @return Number of dimensions.
     */
    static constexpr std::size_t dim() { return grid_type::dim(); }

    /**
     * @brief Construct a linear interpolation object on a given grid with the given function values.
     *
     * @param grid N-dimensional grid.
     * @param fvals Function values at the grid points.
     * @param order Storage order of the multi-dimensional array containing the function values (used
     * only for type deduction).
     */
    linear_interpolation_nd(
        const grid_type& grid, const std::span<double>& fvals, [[maybe_unused]] Order order = Order {}) :
        grid_(grid),
        fvals_(fvals) {
        if (grid.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "linear_interpolation_nd::linear_interplation_nd");
        }
    }

    /**
     * @brief Perform the linear interpolation.
     *
     * @details It recursively performs \f$ 2^N - 1 \f$ linear interpolations, where \f$ N \f$
     * corresponds to the number of dimensions.
     *
     * @param x_arr Value array at which we seek the function value, i.e. \f$ \mathbf{x} \f$.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(const grid_type::nd_value_type& x_arr) const {
        const auto idxs = grid_.index_subrange(2, x_arr);
        const auto xds = detail::distance_ratios(idxs, x_arr, grid_, std::make_index_sequence<dim()>());
        return detail::interp_linear_nd<dim() - 1, grid_type>(idxs, xds, fvals_, grid_.shape(), Order {});
    }

    /**
     * @brief Perform the linear interpolation.
     *
     * @details It recursively performs \f$ 2^N - 1 \f$ linear interpolations, where \f$ N \f$
     * corresponds to the number of dimensions.
     *
     * @tparam Vals Value types.
     * @param xs Values at which we seek the function value, i.e. the \f$ x_i \f$.
     * @return Interpolated value.
     */
    template <typename... Vals>
    [[nodiscard]] double operator()(Vals... xs) const {
        return this->operator()(typename grid_type::nd_value_type { xs... });
    }

    /**
     * @brief Get the N-dimensional grid.
     *
     * @return N-dimensional grid object.
     */
    [[nodiscard]] const auto& grid() const { return grid_; }

    /**
     * @brief Get the function values.
     *
     * @return `std::span` containing the function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    grid_type grid_;
    std::span<double> fvals_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
