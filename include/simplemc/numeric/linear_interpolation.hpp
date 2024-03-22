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

namespace detail {

/**
 * @brief Perform linear interpolation in 1-dimension at point x.
 *
 * @details The value x lies between x0 and x1 with the corresponding function
 * values f0 and f1.
 *
 * @param xd Ratio of distance in x-direction, i.e. (x - x0) / (x1 - x0).
 * @param f0 Function value at x0, i.e. f(x0).
 * @param f1 Function value at x1, i.e. f(x1).
 * @return Interpolated value.
 */
inline double interp_linear_1d(double xd, double f0, double f1) {
    return f0 * (1 - xd) + f1 * xd;
}

/**
 * @brief Perform linear interpolation in n-dimensions.
 *
 * @tparam N Current dimension.
 * @tparam Grid Type of the n-dimensional grid.
 * @tparam Order Index order.
 * @param idx_arr Index array representing the lower left corner of the hypercube in which we interpolate.
 * @param xd_arr Array of ratios of the distances in each direction.
 * @param fvals Span containing the function values.
 * @param shape_arr Shape of the grid (used for indexing the correct function values).
 * @param order Index order of the multi-dimensional array containing the function values.
 * @return Interpolated value.
 */
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

/**
 * @brief Get ratios of the distances in each direction, i.e. (x[i] - x0[i]) / (x1[i] - x0[i]).
 *
 * @tparam Grid Type of the n-dimensional grid.
 * @param idx_arr Index array.
 * @param x_arr Value array.
 * @param grid n-dimensional grid.
 * @return Array containing the ratios of the distances.
 */
template <typename Grid, std::size_t... Is>
inline auto distance_ratios(const typename Grid::nd_size_type& idx_arr, const typename Grid::nd_value_type& x_arr,
    const Grid& grid, std::index_sequence<Is...>) {
    return typename Grid::nd_value_type { (x_arr[Is] - std::get<Is>(grid.grids()).at(idx_arr[Is])) /
        (std::get<Is>(grid.grids()).at(idx_arr[Is] + 1) - std::get<Is>(grid.grids()).at(idx_arr[Is]))... };
}

} // namespace detail

/**
 * @brief 1D linear interpolation.
 *
 * @details The function values are not owned by the interpolation class. Only a span
 * to the original data is stored.
 *
 * @tparam Grid Type of grid in x direction.
 */
template <typename Grid>
class linear_interpolation {
public:
    /**
     * @brief Grid type.
     */
    using grid_type = Grid;

    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions.
     */
    static constexpr std::size_t dim() { return 1; }

    /**
     * @brief Construct a linear interpolation object on a given grid with given function values.
     *
     * @param grid Grid in x direction.
     * @param fvals Function values at grid points.
     */
    linear_interpolation(const grid_type& grid, const std::span<double>& fvals);

    /**
     * @brief Perform interpolation.
     *
     * @param x Value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(double x) const;

    /**
     * @brief Get grid in x direction.
     */
    [[nodiscard]] const auto& grid() const { return grid_; }

    /**
     * @brief Get function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    grid_type grid_;
    std::span<double> fvals_;
};

template <typename Grid>
linear_interpolation<Grid>::linear_interpolation(const grid_type& grid, const std::span<double>& fvals) :
    grid_(grid),
    fvals_(fvals) {
    if (grid.size() != static_cast<grid_type::size_type>(fvals_.size())) {
        throw simplemc_exception(
            "Number of grid points not equal to number of y values.", "linear_interpolation::linear_interplation");
    }
}

template <typename Grid>
double linear_interpolation<Grid>::operator()(double x) const {
    const auto idx = grid_.index_subrange(2, x);
    const auto xd = (x - grid_.at(idx)) / (grid_.at(idx + 1) - grid_.at(idx));
    return detail::interp_linear_1d(xd, fvals_[idx], fvals_[idx + 1]);
}

/**
 * @brief n-dimensional linear interpolation.
 *
 * @details The function values are not owned by the interpolation class. Only a span
 * to the original data is stored.
 *
 * @tparam Grid Type of n-dimensional grid.
 * @tparam Order Index order.
 */
template <typename Grid, nd_order Order = column_major>
class linear_interpolation_nd {
public:
    /**
     * @brief n-dimensional grid type.
     */
    using grid_type = Grid;

    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions.
     */
    static constexpr std::size_t dim() { return grid_type::dim(); }

    /**
     * @brief Construct a linear interpolation object on a given grid with given function values.
     *
     * @param grid n-dimensional grid.
     * @param fvals Function values at grid points.
     * @param order Index order of the multi-dimensional array containing the function values.
     */
    linear_interpolation_nd(
        const grid_type& grid, const std::span<double>& fvals, [[maybe_unused]] Order order = Order {});

    /**
     * @brief Perform interpolation.
     *
     * @param x_arr Value array at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(const grid_type::nd_value_type& x_arr) const;

    /**
     * @brief Perform interpolation.
     *
     * @tparam Vals Value types.
     * @param xs Values at which we seek the function value.
     * @return Interpolated value.
     */
    template <typename... Vals>
    [[nodiscard]] double operator()(Vals... xs) const;

    /**
     * @brief Get n-dimensional grid.
     */
    [[nodiscard]] const auto& grid() const { return grid_; }

    /**
     * @brief Get function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    grid_type grid_;
    std::span<double> fvals_;
};

template <typename Grid, nd_order Order>
linear_interpolation_nd<Grid, Order>::linear_interpolation_nd(
    const grid_type& grid, const std::span<double>& fvals, [[maybe_unused]] Order order) :
    grid_(grid),
    fvals_(fvals) {
    if (grid.size() != static_cast<grid_type::size_type>(fvals_.size())) {
        throw simplemc_exception("Number of grid points not equal to number of y values.",
            "linear_interpolation_nd::linear_interplation_nd");
    }
}

template <typename Grid, nd_order Order>
double linear_interpolation_nd<Grid, Order>::operator()(const grid_type::nd_value_type& x_arr) const {
    const auto idxs = grid_.index_subrange(2, x_arr);
    const auto xds = detail::distance_ratios(idxs, x_arr, grid_, std::make_index_sequence<dim()>());
    return detail::interp_linear_nd<dim() - 1, grid_type>(idxs, xds, fvals_, grid_.shape(), Order {});
}

template <typename Grid, nd_order Order>
template <typename... Vals>
double linear_interpolation_nd<Grid, Order>::operator()(Vals... xs) const {
    return this->operator()(typename grid_type::nd_value_type { xs... });
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
