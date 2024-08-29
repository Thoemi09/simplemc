/**
 * @file
 * @brief Polynomial interpolation of arbitrary degree in arbitrary dimensions.
 */

#ifndef SIMPLEMC_NUMERIC_POLYNOMIAL_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_POLYNOMIAL_INTERPOLATION_HPP

#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/indexing.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-interpolation
 * @{
 */

namespace detail {

// Perform Neville's algorithm given the function values at the grid points, the distances between
// x and the grid points and the number of grid points used, i.e. the degree of polynomial + 1.
// See https://en.wikipedia.org/wiki/Neville%27s_algorithm.
inline double neville(std::vector<double>& fvec, const std::vector<double>& xvec, std::size_t npoints) {
    assert(fvec.size() == npoints);
    assert(xvec.size() == npoints);
    for (std::size_t i = 1; i < npoints; ++i) {
        for (std::size_t j = 0; j < npoints - i; ++j) {
            fvec[j] = (xvec[j + i] * fvec[j] - xvec[j] * fvec[j + 1]) / (xvec[j + i] - xvec[j]);
        }
    }
    return fvec[0];
}

} // namespace detail

/**
 * @brief Polynomial interpolation of arbitrary degree in 1D.
 *
 * @details The class stores a 1-dimensional grid (see @ref simplemc-grids-1d) together with the
 * function values at the grid points.
 *
 * If the degree of the polynomial is \f$ m \f$, the interpolation is done using \f$ m + 1 \f$ grid
 * points and function values.
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
 * auto interp = simplemc::polynomial_interpolation(grid, fvals, 2)
 *
 * // use the interpolation object ...
 * @endcode
 *
 * @tparam Grid 1-dimensional grid type.
 */
template <typename Grid>
class polynomial_interpolation {
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
     * @brief Construct a polynomial interpolation object of a certain degree on a given grid with the
     * given function values.
     *
     * @param grid 1-dimensional grid on the \f$ x \f$ axis.
     * @param fvals Function values at grid points.
     * @param degree Degree of polynomial.
     */
    polynomial_interpolation(const grid_type& grid, const std::span<double>& fvals, std::size_t degree) :
        grid_(grid),
        fvals_(fvals),
        npoints_(degree + 1),
        fvec_(npoints_, 0.0),
        xvec_(npoints_, 0.0) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "polynomial_interpolation::polynomial_interpolation");
        }
        if (grid_.size() < static_cast<grid_type::size_type>(npoints_)) {
            throw simplemc_exception(
                "Degree of polynomial too high", "polynomial_interpolation::polynomial_interpolation");
        }
    }

    /**
     * @brief Perform the polynomial interpolation.
     *
     * @param x Value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(double x) const {
        auto idx = grid_.index_subrange(npoints_, x);
        for (std::size_t i = 0; i < npoints_; ++i) {
            fvec_[i] = fvals_[idx + i];
            xvec_[i] = x - grid_.at(idx + i);
        }
        return detail::neville(fvec_, xvec_, npoints_);
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

    /**
     * @brief Get the degree of the polynomial.
     *
     * @return Degree of the interpolating polynomial.
     */
    [[nodiscard]] auto degree() const { return npoints_ - 1; }

private:
    grid_type grid_;
    std::span<double> fvals_;
    std::size_t npoints_;
    mutable std::vector<double> fvec_;
    mutable std::vector<double> xvec_;
};

/**
 * @brief Polynomial interpolation of arbitrary degree in \f$ N \f$ dimensions.
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
class polynomial_interpolation_nd {
public:
    /**
     * @brief  N-dimensional grid type.
     */
    using grid_type = Grid;

    /**
     * @brief Get the number of dimensions.
     *
     * @return Number of dimensions.
     */
    static constexpr std::size_t dim() { return grid_type::dim(); }

    /**
     * @brief Construct a polynomial interpolation object of a certain degree on a given grid with the
     * given function values.
     *
     * @param grid N-dimensional grid.
     * @param fvals Function values at grid points.
     * @param degree Degree of polynomial.
     * @param order Storage order of the multi-dimensional array containing the function values (used
     * only for type deduction).
     */
    polynomial_interpolation_nd(const grid_type& grid, const std::span<double>& fvals, std::size_t degree,
        [[maybe_unused]] Order order = Order {}) :
        grid_(grid),
        fvals_(fvals),
        npoints_(degree + 1) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "polynomial_interpolation_nd::polynomial_interpolation_nd");
        }
        if (std::apply(
                [this](const auto... x) { return ((x.size() < static_cast<grid_type::size_type>(npoints_)) && ...); },
                grid_.grids())) {
            throw simplemc_exception(
                "Degree of polynomial too high", "polynomial_interpolation_nd::polynomial_interpolation_nd");
        }
        for (std::size_t i = 0; i < dim(); ++i) {
            fvecs_[i].resize(npoints_);
            xvecs_[i].resize(npoints_);
        }
    }

    /**
     * @brief Perform the polynomial interpolation.
     *
     * @details It recursively performs \f$ 2^N - 1 \f$ polynomial interpolations, where \f$ N \f$
     * corresponds to the number of dimensions.
     *
     * @param x_arr Value array at which we seek the function value, i.e. \f$ \mathbf{x} \f$.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(const grid_type::nd_value_type& x_arr) const {
        const auto idx_arr = grid_.index_subrange(npoints_, x_arr);
        return interp_poly_nd<dim() - 1>(x_arr, idx_arr, grid_.shape());
    }

    /**
     * @brief Perform the polynomial interpolation.
     *
     * @details It recursively performs \f$ 2^N - 1 \f$ polynomial interpolations, where \f$ N \f$
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

    /**
     * @brief Get the degree of the polynomial.
     *
     * @return Degree of the interpolating polynomial.
     */
    [[nodiscard]] auto degree() const { return npoints_ - 1; }

private:
    // Perform polynomial interpolation in N-dimensions given the value array at which we seek the
    // function value (the vector x), the index array representing the lower left corner of the
    // hypercube in which we interpolate and the shape of the grid.
    template <int N>
    double interp_poly_nd(const typename grid_type::nd_value_type& x_arr, const typename Grid::nd_size_type& idx_arr,
        const typename Grid::nd_size_type& shape_arr) const {
        static_assert(N >= 0 && N < Grid::dim(), "Invalid template parameter in interp_poly_nd.");
        auto idx_p_arr = idx_arr;
        const auto& grid_n = std::get<N>(grid_.grids());
        if constexpr (N == 0) {
            // perform simple polynomial interpolation in the first dimension
            for (std::size_t i = 0; i < npoints_; ++i) {
                idx_p_arr[N] = idx_arr[N] + i;
                fvecs_[N][i] = fvals_[flat_index(idx_p_arr, shape_arr, Order {})];
                xvecs_[N][i] = x_arr[N] - grid_n.at(idx_p_arr[N]);
            }
            return detail::neville(fvecs_[N], xvecs_[N], npoints_);
        } else {
            // perform polynomial interpolation in N-1 dimensions
            for (std::size_t i = 0; i < npoints_; ++i) {
                idx_p_arr[N] = idx_arr[N] + i;
                fvecs_[N][i] = interp_poly_nd<N - 1>(x_arr, idx_p_arr, shape_arr);
                xvecs_[N][i] = x_arr[N] - grid_n.at(idx_p_arr[N]);
            }
            return detail::neville(fvecs_[N], xvecs_[N], npoints_);
        }
    }

    grid_type grid_;
    std::span<double> fvals_;
    std::size_t npoints_;
    mutable std::array<std::vector<double>, dim()> fvecs_;
    mutable std::array<std::vector<double>, dim()> xvecs_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_POLYNOMIAL_INTERPOLATION_HPP
