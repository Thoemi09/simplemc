/**
 * @file
 * @brief Linear interpolation in arbitrary dimensions.
 */

#ifndef SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP

#include <simplemc/grids/concepts.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/nd_indexing.hpp>
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
[[nodiscard]] constexpr double interp_linear_1d(double xd, double f0, double f1) noexcept {
    return f0 * (1 - xd) + f1 * xd;
}

} // namespace detail

/**
 * @brief Linear interpolation in 1D.
 *
 * @details The linear interpolant \f$ h(x) \f$ of the function \f$ f(x) \f$ is defined as the
 * piecewise linear function
 * \f[
 *   h(x) =
 *   \begin{cases}
 *     h_0(x) &= a_1 + b_1 x & \text{if } x \in [x_0, x_1] \\
 *     h_1(x) &= a_2 + b_2 x & \text{if } x \in [x_1, x_2] \\
 *     \vdots \\
 *     h_{M-2}(x) &= a_{M-2} + b_{M-2} x & \text{if } x \in [x_{M-2}, x_{M-1}] \\
 *   \end{cases}
 *   \; ,
 * \f]
 * where \f$ h_i(x) \f$ is the straight line that passes through the two function values \f$ f_i =
 * f(x_i) \f$ and \f$ f_{i+1} = f(x_{i+1}) \f$ at the points \f$ x_i = g(i) \f$ and \f$ x_{i+1} =
 * g(i+1) \f$:
 * \f[
 *   h_i(x) = f_i + (x - x_i) \frac{f_{i+1} - f_i}{x_{i+1} - x_i} = \frac{f_i (x_{i+1} - x) + f_{i+1}
 *   (x - x_i)}{x_{i+1} - x_i} \; .
 * \f]
 *
 * The class stores
 * - a simplemc::grid_1d \f$ g \f$ (see @ref simplemc-grids-1d) that determines the grid points \f$
 * g(i) = x_i \f$ and
 * - the function values \f$ f_i = f(g(i)) \f$ at the grid points \f$ g(i) \f$.
 *
 * @note The function values are not owned by the interpolation class. Only a `std::span` to the
 * original data is stored.
 *
 * @include numeric/doc_linear_interpolation_1d.cpp
 *
 * Output:
 *
 * ```
 * x         interp(x)           f(x)
 * 0.0       0                   0
 * 0.2       0.040149938         0.04
 * 0.4       0.16026656          0.16
 * 0.6       0.36034985          0.36
 * 0.8       0.64039983          0.64
 * 1.0       1.0004165           1
 * 1.2       1.4403998           1.44
 * 1.4       1.9603499           1.96
 * 1.6       2.5602666           2.56
 * 1.8       3.2401499           3.24
 * 2.0       4                   4
 * ```
 *
 * @tparam Grid simplemc::grid_1d grid type.
 */
template <grid_1d Grid>
class linear_interpolation {
public:
    /**
     * @brief 1-dimensional grid type (see simplemc::grid_1d).
     */
    using grid_type = Grid;

    /**
     * @brief Get the dimensionality of the domain \f$ \mathrm{D} \subset \mathbb{R}^1 \f$.
     *
     * @return Number of dimensions, \f$ N = 1 \f$.
     */
    [[nodiscard]] static constexpr std::size_t dim() noexcept { return 1; }

    /**
     * @brief Construct a linear interpolation object on a given grid \f$ g \f$ with the given
     * function values \f$ f_i = f(x_i) = f(g(i)) \f$.
     *
     * @details It throws a simplemc::simplemc_exception if the size of the grid \f$ M \f$ is not
     * equal to the number of function values.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g 1-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_i \f$.
     */
    constexpr linear_interpolation(grid_type g, const std::span<double>& fvals) : grid_(std::move(g)), fvals_(fvals) {
        if (grid_.size() != static_cast<long>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values");
        }
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ x \f$.
     *
     * @details The given point \f$ x \f$ is assumed to lie in the domain of the function \f$ f \f$.
     * If this is not the case, the behaviour is undefined.
     *
     * The value of the interpolant at \f$ x \f$ is given by
     * \f[
     *   h(x) = f_i + (x - x_i) \frac{f_{i+1} - f_i}{x_{i+1} - x_i} \; ,
     * \f]
     * where \f$ x \in [x_i, x_{i+1}] \f$.
     *
     * @param x Point \f$ x \in \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(x) \f$.
     */
    [[nodiscard]] constexpr double operator()(double x) const {
        const auto idx = index_subrange(grid_, 2, x);
        const auto xd = (x - grid_.at(idx)) / (grid_.at(idx + 1) - grid_.at(idx));
        return detail::interp_linear_1d(xd, fvals_[idx], fvals_[idx + 1]);
    }

    /**
     * @brief Get the underlying grid \f$ g \f$ on which the function values \f$ f_i \f$ are evaluated.
     *
     * @return 1-dimensional grid \f$ g \f$.
     */
    [[nodiscard]] constexpr const auto& grid() const noexcept { return grid_; }

    /**
     * @brief Get the function values \f$ f_i = f(g(i)) \f$.
     *
     * @return `std::span` containing the function values \f$ f_i \f$ at the grid points \f$ g(i) \f$.
     */
    [[nodiscard]] constexpr const auto& function_values() const noexcept { return fvals_; }

private:
    grid_type grid_;
    std::span<double> fvals_;
};

/**
 * @brief Linear interpolation in \f$ N \f$ dimensions.
 *
 * @details We implement linear interpolation in \f$ N \f$ dimensions in terms of \f$ 2^N - 1 \f$
 * 1-dimensional interpolations (see simplemc::linear_interpolation).
 *
 * For example, the algorithm to interpolate a function \f$ f : \mathrm{D} \subset \mathbb{R}^2 \to
 * \mathbb{R} \f$ at a value \f$ \mathbf{x} = (x, y) \in \mathrm{D} \f$ in 2D is as follows:
 * - Find the bin volume \f$ b_{i,j} = [x_i, x_{i+1}) \times [y_j, y_{j+1}) \f$ of the 2-dimensional
 * grid \f$ g \f$ such that \f$ \mathbf{x} \in b_{i,j} \f$.
 * - Interpolate 2 times along the first dimension (x-axis) for fixed y-values \f$ y = y_j \f$ and \f$
 * y = y_{j+1} \f$, respectively:
 * \f{eqnarray*}{
 *   h_{y_j}(x) &=& f(x_i, y_j) + (x - x_i) \frac{f(x_{i+1}, y_j) - f(x_i, y_j)}{x_{i+1} - x_i} \;, \\
 *   h_{y_{j+1}}(x) &=& f(x_i, y_{j+1}) + (x - x_i) \frac{f(x_{i+1}, y_{j+1}) - f(x_i, y_{j+1})}
 *   {x_{i+1} - x_i}
 * \f}
 * - Interpolate a thrid and last time along the second dimension (y-axis) between the two values \f$
 * h_{y_j}(x) \f$ and \f$ h_{y_{j+1}}(x) \f$:
 * \f[
 *   h(y) = h_{y_j}(x) + (y - y_j) \frac{h_{y_{j+1}}(x) - h_{y_j}(x)}{y_{j+1} - y_j} \; .
 * \f]
 *
 * Generalizations to higher dimensions are straightforward.
 *
 * The class stores
 * - a simplemc::grid_nd \f$ g \f$ (see @ref simplemc-grids-nd) that determines the grid points \f$
 * g(i_1, \dots, i_N) = (x_1, \dots, x_N) \f$ and
 * - the function values \f$ f_{i_1, \cdots, i_N} = f(x_1, \dots, x_N) \f$ at the grid points \f$
 * g(i_1, \dots, i_N) \f$.
 *
 * The second template parameter describes the memory layout of the multi-dimensional array containing
 * the function values (see simplemc::row_major and simplemc::column_major). By default, column-major
 * order is assumed.
 *
 * @note The function values are not owned by the interpolation class. Only a `std::span` to the
 * original data is stored.
 *
 * @include numeric/doc_linear_interpolation_nd.cpp
 *
 * Output:
 *
 * ```
 * (x, y)         interp(x, y)        f(x, y)
 * (0,0)          0                   0
 * (0,0.2)        0.040149938         0.04
 * (0,0.4)        0.16026656          0.16
 * (0,0.6)        0.36034985          0.36
 * //             //                  //
 * (2,1.4)        5.9603499           5.96
 * (2,1.6)        6.5602666           6.56
 * (2,1.8)        7.2401499           7.24
 * (2,2)          8                   8
 * ```
 *
 * @tparam Grid simplemc::grid_nd grid type.
 * @tparam Order Storage order of the multi-dimensional array containing the function values.
 */
template <grid_nd Grid, nd_order Order = column_major>
class linear_interpolation_nd {
public:
    /**
     * @brief N-dimensional grid type (see simplemc::grid_nd).
     */
    using grid_type = Grid;

    /**
     * @brief Get the dimensionality of the domain \f$ \mathrm{D} \subset \mathbb{R}^N \f$.
     *
     * @return Number of dimensions, \f$ N \f$.
     */
    [[nodiscard]] static constexpr std::size_t dim() noexcept { return grid_type::dim(); }

    /**
     * @brief Construct a linear interpolation object on a given grid \f$ g \f$ with the given
     * function values \f$ f_{i_1, \cdots, i_N} = f(x_1, \dots, x_n) = f(g(i_1, \dots, i_N)) \f$.
     *
     * @details It throws a simplemc::simplemc_exception if the size of the grid \f$ M \f$ is not
     * equal to the number of function values.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g N-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_{i_1, \cdots, i_N} \f$.
     * @param order Storage order of the multi-dimensional array containing the function values (used
     * only for type deduction).
     */
    constexpr linear_interpolation_nd(
        grid_type g, const std::span<double>& fvals, [[maybe_unused]] Order order = Order {}) :
        grid_(std::move(g)),
        fvals_(fvals) {
        if (grid_.size() != static_cast<long>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values");
        }
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ \mathbf{x} = (x_1, \dots, x_N) \f$.
     *
     * @details The given point \f$ \mathbf{x} \f$ is assumed to lie in the domain of the function
     * \f$ f \f$. If this is not the case, the behaviour is undefined.
     *
     * It recursively performs \f$ 2^N - 1 \f$ 1-dimensional linear interpolations, where \f$ N \f$
     * corresponds to the number of dimensions.
     *
     * @param x Point \f$ \mathbf{x} \in \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(\mathbf{x}) \f$.
     */
    [[nodiscard]] constexpr double operator()(const grid_type::value_type& x) const {
        const auto idxs = index_subrange(grid_, 2, x);
        const auto xds = distance_ratios(idxs, x, std::make_index_sequence<dim()>());
        return interp_linear_nd<dim() - 1>(idxs, xds);
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ \mathbf{x} = (x_1, \dots, x_N) \f$.
     *
     * @details The given point \f$ \mathbf{x} \f$ is assumed to lie in the domain of the function
     * \f$ f \f$. If this is not the case, the behaviour is undefined.
     *
     * It recursively performs \f$ 2^N - 1 \f$ 1-dimensional linear interpolations, where \f$ N \f$
     * corresponds to the number of dimensions.
     *
     * @tparam Vals Value types.
     * @param xs Values \f$ x_1, \dots, x_N \f$ of the point \f$ \mathbf{x} = (x_1, \dots, x_N) \in
     * \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(\mathbf{x}) \f$.
     */
    template <typename... Vals>
    [[nodiscard]] constexpr double operator()(Vals... xs) const {
        return this->operator()(typename grid_type::value_type { xs... });
    }

    /**
     * @brief Get the underlying grid \f$ g \f$ on which the function values \f$ f_{i_1, \cdots, i_N}
     * \f$ are evaluated.
     *
     * @return N-dimensional grid \f$ g \f$.
     */
    [[nodiscard]] constexpr const auto& grid() const noexcept { return grid_; }

    /**
     * @brief Get the function values \f$ f_{i_1, \cdots, i_N} = f(g(i_1, \dots, i_N)) \f$.
     *
     * @return `std::span` containing the function values \f$ f_{i_1, \cdots, i_N} \f$ at the grid
     * points \f$ g(i_1, \dots, i_N) \f$.
     */
    [[nodiscard]] constexpr const auto& function_values() const noexcept { return fvals_; }

private:
    // Linear interpolation given the index array of the bin at the lower left corner of the volume in
    // which to interpolate and the array of ratios of the distances in each direction (see xd in
    // interp_linear_1d and distance_ratios below).
    //
    // This function is called recursively, with N being the current dimension.
    template <int N>
    [[nodiscard]] constexpr double interp_linear_nd(
        const typename grid_type::size_type& idx_arr, const typename grid_type::value_type& xd_arr) const {
        static_assert(N >= 0 && N < grid_type::dim(), "Invalid template parameter in interp_linear_nd.");
        auto idx_p1_arr = idx_arr;
        idx_p1_arr[N] += 1;
        if constexpr (N == 0) {
            // perform simple linear interpolation in the first dimension
            const auto f0 = fvals_[flat_index(idx_arr, grid_.shape(), Order {})];
            const auto f1 = fvals_[flat_index(idx_p1_arr, grid_.shape(), Order {})];
            return detail::interp_linear_1d(xd_arr[N], f0, f1);
        } else {
            // perform linear interpolation in n-1 dimensions
            const auto f0 = interp_linear_nd<N - 1>(idx_arr, xd_arr);
            const auto f1 = interp_linear_nd<N - 1>(idx_p1_arr, xd_arr);
            return detail::interp_linear_1d(xd_arr[N], f0, f1);
        }
    }

    // Get ratios of the distances in each direction, i.e. (x[i] - x0[i]) / (x1[i] - x0[i]), given the
    // index array of the bin at the lower left corner of the volume in which to interpolate and the
    // point x.
    template <std::size_t... Is>
    [[nodiscard]] constexpr auto distance_ratios(const typename grid_type::size_type& idx_arr,
        const typename grid_type::value_type& x, std::index_sequence<Is...>) const {
        return typename grid_type::value_type { (x[Is] - std::get<Is>(grid_.grids()).at(idx_arr[Is])) /
            (std::get<Is>(grid_.grids()).at(idx_arr[Is] + 1) - std::get<Is>(grid_.grids()).at(idx_arr[Is]))... };
    }

private:
    grid_type grid_;
    std::span<double> fvals_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
