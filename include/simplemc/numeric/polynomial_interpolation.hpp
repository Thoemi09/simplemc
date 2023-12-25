/**
 * @file polynomial_interpolation.hpp
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

namespace detail {

/**
 * @brief Perform Neville's algorithm.
 *
 * @details See https://en.wikipedia.org/wiki/Neville%27s_algorithm.
 *
 * @param fvec Function values.
 * @param xvec Distances of x to the grid points x_i.
 * @param npoints Number of points used, i.e. degree of polynomial + 1.
 */
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
 * @brief 1D polynomial interpolation of arbitrary degree.
 *
 * @details The function values are not owned by the interpolation class. Only a span
 * to the original data is stored.
 *
 * @tparam Grid Type of grid.
 */
template <typename Grid>
class polynomial_interpolation {
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
     * @brief Construct a polynomial interpolation object of a certain degree on a given grid with
     * given function values.
     *
     * @param grid Grid in x direction.
     * @param fvals Function values at grid points.
     * @param degree Degree of polynomial.
     */
    polynomial_interpolation(const grid_type& grid, const std::span<double>& fvals, std::size_t degree);

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
    std::size_t npoints_;
    mutable std::vector<double> fvec_;
    mutable std::vector<double> xvec_;
};

template <typename Grid>
polynomial_interpolation<Grid>::polynomial_interpolation(
    const grid_type& grid, const std::span<double>& fvals, std::size_t degree) :
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
        throw simplemc_exception("Degree of polynomial too high", "polynomial_interpolation::polynomial_interpolation");
    }
}

template <typename Grid>
double polynomial_interpolation<Grid>::operator()(double x) const {
    auto idx = grid_.index_subrange(npoints_, x);
    for (std::size_t i = 0; i < npoints_; ++i) {
        fvec_[i] = fvals_[idx + i];
        xvec_[i] = x - grid_.at(idx + i);
    }
    return detail::neville(fvec_, xvec_, npoints_);
}

/**
 * @brief n-dimensional polynomial interpolation of arbitrary degree.
 *
 * @details The function values are not owned by the interpolation class. Only a span
 * to the original data is stored.
 *
 * @tparam Grid Type of n-dimensional grid.
 * @tparam Order Index order.
 */
template <typename Grid, nd_order Order = column_major>
class polynomial_interpolation_nd {
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
    static constexpr std::size_t dim() { return grid_type::dim(); }

    /**
     * @brief Construct a polynomical interpolation object on a given grid with given function values.
     *
     * @param grid n-dimensional grid.
     * @param fvals Function values at grid points.
     * @param degree Degree of polynomial.
     * @param order Order of the multi-dimensional array containing the function values.
     */
    polynomial_interpolation_nd(const grid_type& grid, const std::span<double>& fvals, std::size_t degree,
        [[maybe_unused]] Order order = Order {});

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
    /**
     * @brief Perform polynomial interpolation in n-dimensions.
     *
     * @tparam N Current dimension.
     * @param x_arr Value array at which we seek the function value.
     * @param idx_arr Index array of the lower left corner of the hypercube in which we interpolate.
     * @param shape_arr Shape of the grid (used for indexing the correct function values).
     * @return Interpolated value.
     */
    template <int N>
    double interp_poly_nd(const typename grid_type::nd_value_type& x_arr, const typename Grid::nd_size_type& idx_arr,
        const typename Grid::nd_size_type& shape_arr) const;

    grid_type grid_;
    std::span<double> fvals_;
    std::size_t npoints_;
    mutable std::array<std::vector<double>, dim()> fvecs_;
    mutable std::array<std::vector<double>, dim()> xvecs_;
};

template <typename Grid, nd_order Order>
polynomial_interpolation_nd<Grid, Order>::polynomial_interpolation_nd(
    const grid_type& grid, const std::span<double>& fvals, std::size_t degree, [[maybe_unused]] Order order) :
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

template <typename Grid, nd_order Order>
double polynomial_interpolation_nd<Grid, Order>::operator()(const grid_type::nd_value_type& x_arr) const {
    const auto idx_arr = grid_.index_subrange(npoints_, x_arr);
    return interp_poly_nd<dim() - 1>(x_arr, idx_arr, grid_.shape());
}

template <typename Grid, nd_order Order>
template <typename... Vals>
double polynomial_interpolation_nd<Grid, Order>::operator()(Vals... xs) const {
    return this->operator()(typename grid_type::nd_value_type { xs... });
}

template <typename Grid, nd_order Order>
template <int N>
double polynomial_interpolation_nd<Grid, Order>::interp_poly_nd(const typename grid_type::nd_value_type& x_arr,
    const typename grid_type::nd_size_type& idx_arr, const typename grid_type::nd_size_type& shape_arr) const {
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
        // perform polynomial interpolation in n-1 dimensions
        for (std::size_t i = 0; i < npoints_; ++i) {
            idx_p_arr[N] = idx_arr[N] + i;
            fvecs_[N][i] = interp_poly_nd<N - 1>(x_arr, idx_p_arr, shape_arr);
            xvecs_[N][i] = x_arr[N] - grid_n.at(idx_p_arr[N]);
        }
        return detail::neville(fvecs_[N], xvecs_[N], npoints_);
    }
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_POLYNOMIAL_INTERPOLATION_HPP
