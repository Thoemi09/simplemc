/**
 * @file
 * @brief Polynomial interpolation of arbitrary degree in arbitrary dimensions.
 */

#ifndef SIMPLEMC_NUMERIC_POLYNOMIAL_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_POLYNOMIAL_INTERPOLATION_HPP

#include <simplemc/grids/concepts.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/nd_indexing.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <utility>
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
 * @details The polynomial interpolant \f$ h(x) \f$ of degree \f$ m \f$ of the function \f$ f(x) \f$
 * is defined as the piecewise function
 * \f[
 *   h(x) =
 *   \begin{cases}
 *     h_0(x) &= a_{0,0} + a_{0,1} x + \dots + a_{0,m} x^m & \text{if } x \in [x_0, x_1] \\
 *     h_1(x) &= a_{1,0} + a_{1,1} x + \dots + a_{1,m} x^m & \text{if } x \in [x_1, x_2] \\
 *     \vdots \\
 *     h_{M-2}(x) &= a_{M-2,0} + a_{M-2,1} x + \dots + a_{M-2,m} x^m & \text{if } x \in [x_{M-2},
 *     x_{M-1}] \\
 *   \end{cases}
 *   \; ,
 * \f]
 * where \f$ h_i(x) \f$ is the unique degree \f$ m \f$ polynomial that passes through the \f$ m + 1
 * \f$ function values \f$ f_{i_j} = f(x_{i_j}), \dots, f_{i_j+m} = f(x_{i_j+m}) \f$ at the points \f$
 * x_{i_j} = g(i_j), \dots, x_{i_j+m} = g(i_j+m) \f$.
 *
 * The function values \f$ f_{i_j}, \dots, f_{i_j+m} \f$ that are used to find the coefficients \f$
 * a_{i,0}, \dots, a_{i,m} \f$ of \f$ h_i(x) \f$ depend on \f$ i \f$, \f$ m \f$ and \f$ M \f$. In
 * practice, we first determine the index \f$ i \f$ and then use simplemc::integer_subrange to obtain
 * the index \f$ i_j \f$ for the given values of \f$ i \f$, \f$ m \f$ and \f$ M \f$. Once we have
 * determined the function values, we apply Neville's algorithm to get the coefficients \f$ a_{i,0},
 * \dots, a_{i,m} \f$.
 *
 * The class stores
 * - a simplemc::grid_1d \f$ g \f$ (see @ref simplemc-grids-1d) that determines the grid points \f$
 * g(i) = x_i \f$,
 * - the function values \f$ f_i = f(g(i)) \f$ at the grid points \f$ g(i) \f$ and
 * - the degree \f$ m \f$.
 *
 * @note The function values are not owned by the interpolation class. Only a `std::span` to the
 * original data is stored.
 *
 * @code{.cpp}
 * #include <fmt/base.h>
 * #include <simplemc/grids.hpp>
 * #include <simplemc/numeric/polynomial_interpolation.hpp>
 * #include <simplemc/utils/ranges.hpp>
 *
 * #include <vector>
 *
 * int main() {
 *     // define the function f(x) = x^3 that we want to interpolate
 *     auto f = [](double x) { return x * x * x; };
 *
 *     // create a linear interpolation grid of size 50 on [0, 2]
 *     simplemc::linear_grid interp_grid { 0, 2, 50 };
 *
 *     // obtain the function values at the grid points
 *     auto fview = simplemc::ranges::transform_view(simplemc::grid_view(interp_grid), [&f](double x) { return f(x); });
 *     auto fvals = std::vector<double>(fview.begin(), fview.end());
 *
 *     // create the interpolation object for f(x) on the interpolation grid
 *     auto interp = simplemc::polynomial_interpolation { interp_grid, fvals, 2 };
 *
 *     // create the grid at which we want to test our interpolation
 *     simplemc::linear_grid test_grid { 0, 2, 11 };
 *
 *     // test the interpolation of the function f(x)
 *     fmt::println("{:<10}{:<20}{:<20}", "x", "interp(x)", "f(x)");
 *     for (auto x : simplemc::grid_view(test_grid)) {
 *         fmt::println("{:<10.1f}{:<20.8g}{:<20.8g}", x, interp(x), f(x));
 *     }
 *     fmt::println("");
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * x         interp(x)           f(x)
 * 0.0       -0                  0
 * 0.2       0.0080116278        0.008
 * 0.4       0.064019584         0.064
 * 0.6       0.21602428          0.216
 * 0.8       0.51202611          0.512
 * 1.0       1.0000255           1
 * 1.2       1.7280228           1.728
 * 1.4       2.7440186           2.744
 * 1.6       4.0960131           4.096
 * 1.8       5.8320067           5.832
 * 2.0       8                   8
 * ```
 *
 * @tparam Grid simplemc::grid_1d grid type.
 */
template <grid_1d Grid>
class polynomial_interpolation {
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
    static constexpr std::size_t dim() { return 1; }

    /**
     * @brief Construct a polynomial interpolation object of degree \f$ m \f$ on a given grid \f$ g
     * \f$ with the given function values \f$ f_i = f(x_i) = f(g(i)) \f$.
     *
     * @details It throws a simplemc::simplemc_exception if the size of the grid \f$ M \f$
     * - is not equal to the number of function values or
     * - is smaller than \f$ m + 1 \f$, i.e. the number of points used for the interpolating
     * polynomials \f$ h_i(x) \f$.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g 1-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_i \f$.
     * @param m Degree \f$ m \f$ of the interpolating polynomials \f$ h_i(x) \f$.
     */
    polynomial_interpolation(grid_type g, const std::span<double>& fvals, std::size_t m) :
        grid_(std::move(g)),
        fvals_(fvals),
        npoints_(m + 1),
        fvec_(npoints_, 0.0),
        xvec_(npoints_, 0.0) {
        if (grid_.size() != static_cast<long>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "polynomial_interpolation::polynomial_interpolation");
        }
        if (grid_.size() < static_cast<long>(npoints_)) {
            throw simplemc_exception(
                "Degree of interpolating polynomial is too high", "polynomial_interpolation::polynomial_interpolation");
        }
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ x \f$.
     *
     * @details The given point \f$ x \f$ is assumed to lie in the domain of the function \f$ f \f$.
     * If this is not the case, the behaviour is undefined.
     *
     * We first call simplemc::index_subrange to find the \f$ m + 1 \f$ grid points \f$ g(i_j), \dots,
     * g(i_j+m) \f$ and corresponding function values \f$ f_{i_j}, \dots, f_{i_j+m} \f$ that we want
     * to use for the interpolation.
     *
     * Then we construct the polynomial interpolant \f$ h_i(x) = a_{i,0} + a_{i,1s} x + \dots +
     * a_{i,m} x^m \f$ using Neville's algorithm.
     *
     * @param x Point \f$ x \in \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(x) \f$.
     */
    [[nodiscard]] double operator()(double x) const {
        auto idx = index_subrange(grid_, npoints_, x);
        for (std::size_t i = 0; i < npoints_; ++i) {
            fvec_[i] = fvals_[idx + i];
            xvec_[i] = x - grid_.at(idx + i);
        }
        return detail::neville(fvec_, xvec_, npoints_);
    }

    /**
     * @brief Get the underlying grid \f$ g \f$ on which the function values \f$ f_i \f$ are evaluated.
     *
     * @return 1-dimensional grid \f$ g \f$.
     */
    [[nodiscard]] const auto& grid() const { return grid_; }

    /**
     * @brief Get the function values \f$ f_i = f(g(i)) \f$.
     *
     * @return `std::span` containing the function values \f$ f_i \f$ at the grid points \f$ g(i) \f$.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

    /**
     * @brief Get the degree \f$ m \f$ of the polynomial.
     *
     * @return Degree of the interpolating polynomials \f$ h_i(x) \f$.
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
 * @details Similar to simplemc::linear_interpolation_nd, we implement polynomial interpolation of
 * degree \f$ m \f$ in \f$ N \f$ dimensions in terms of \f$ 2^N - 1 \f$ 1-dimensional polynomial
 * interpolations of degree \f$ m \f$ (see simplemc::polynomial_interpolation).
 *
 * The class stores
 * - a simplemc::grid_nd \f$ g \f$ (see @ref simplemc-grids-nd) that determines the grid points \f$
 * g(i_1, \dots, i_N) = (x_1, \dots, x_N) \f$,
 * - the function values \f$ f_{i_1, \cdots, i_N} = f(x_1, \dots, x_N) \f$ at the grid points \f$
 * g(i_1, \dots, i_N) \f$ and
 * - the degree \f$ m \f$.
 *
 * The second template parameter describes the memory layout of the multi-dimensional array containing
 * the function values (see simplemc::row_major and simplemc::column_major). By default, column-major
 * order is assumed.
 *
 * @note The function values are not owned by the interpolation class. Only a `std::span` to the
 * original data is stored.
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/grids.hpp>
 * #include <simplemc/numeric/polynomial_interpolation.hpp>
 * #include <simplemc/utils/nd_indexing.hpp>
 * #include <simplemc/utils/ranges.hpp>
 *
 * #include <vector>
 *
 * int main() {
 *     // define the function f(x, y) = x^2 + y^3 that we want to interpolate
 *     auto f = [](double x, double y) { return x * x + y * y * y; };
 *
 *     // create a linear 2D interpolation grid of size 50 x 50 on [0, 2] x [0, 2]
 *     simplemc::linear_grid grid_1d { 0, 2, 50 };
 *     simplemc::nd_grid interp_grid { grid_1d, grid_1d };
 *
 *     // obtain the function values at the grid points (in row-major order)
 *     auto fview = simplemc::ranges::transform_view(
 *         simplemc::grid_view(interp_grid), [&f](const auto& x) { return f(x[0], x[1]); });
 *     auto fvals = std::vector<double>(fview.begin(), fview.end());
 *
 *     // create the interpolation object for f(x, y) on the interpolation grid
 *     auto interp = simplemc::polynomial_interpolation_nd { interp_grid, fvals, 2, simplemc::row_major {} };
 *
 *     // create the grid at which we want to test our interpolation
 *     grid_1d.reset(0, 2, 11);
 *     simplemc::nd_grid test_grid { grid_1d, grid_1d };
 *
 *     // test the interpolation of the function f(x, y)
 *     fmt::println("{:<15}{:<20}{:<20}", "(x, y)", "interp(x, y)", "f(x, y)");
 *     for (auto [x, y] : simplemc::grid_view(test_grid)) {
 *         fmt::print("{:<15}{:<20.8g}{:<20.8g}\n", fmt::format("({:.1g},{:.1g})", x, y), interp(x, y), f(x, y));
 *     }
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * (x, y)         interp(x, y)        f(x, y)
 * (0,0)          -0                  0
 * (0,0.2)        0.0080116278        0.008
 * (0,0.4)        0.064019584         0.064
 * (0,0.6)        0.21602428          0.216
 * //             //                  //
 * (2,1)          6.7440186           6.744
 * (2,2)          8.0960131           8.096
 * (2,2)          9.8320067           9.832
 * (2,2)          12                  12
 * ```
 *
 * @tparam Grid simplemc::grid_nd grid type.
 * @tparam Order Storage order of the multi-dimensional array containing the function values.
 */
template <grid_nd Grid, nd_order Order = column_major>
class polynomial_interpolation_nd {
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
    static constexpr std::size_t dim() { return grid_type::dim(); }

    /**
     * @brief Construct a polynomial interpolation object of degree \f$ m \f$ on a given grid \f$ g
     * \f$ with the given function values \f$ f_{i_1, \cdots, i_N} = f(x_1, \dots, x_n) = f(g(i_1,
     * \dots, i_N)) \f$.
     *
     * It throws a simplemc::simplemc_exception
     * - if the size of the grid \f$ M \f$ is not equal to the number of function values or
     * - if the size of one of the underlying 1-dimensional grids \f$ M_k \f$ is smaller than
     * \f$ m + 1 \f$, i.e. the number of points used for the interpolating polynomials \f$ h_i(x_i)
     * \f$ in 1D.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g N-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_{i_1, \cdots, i_N} \f$.
     * @param m Degree \f$ m \f$ of the interpolating polynomials \f$ h_i(x_i) \f$ in 1D.
     * @param order Storage order of the multi-dimensional array containing the function values (used
     * only for type deduction).
     */
    polynomial_interpolation_nd(
        grid_type g, const std::span<double>& fvals, std::size_t m, [[maybe_unused]] Order order = Order {}) :
        grid_(std::move(g)),
        fvals_(fvals),
        npoints_(m + 1) {
        if (grid_.size() != static_cast<long>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "polynomial_interpolation_nd::polynomial_interpolation_nd");
        }
        if (std::apply([this](const auto... xs) { return ((xs.size() < static_cast<long>(npoints_)) && ...); },
                grid_.grids())) {
            throw simplemc_exception("Degree of interpolating polynomial is too high",
                "polynomial_interpolation_nd::polynomial_interpolation_nd");
        }
        for (std::size_t i = 0; i < dim(); ++i) {
            fvecs_[i].resize(npoints_);
            xvecs_[i].resize(npoints_);
        }
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ \mathbf{x} = (x_1, \dots, x_N) \f$.
     *
     * @details The given point \f$ \mathbf{x} \f$ is assumed to lie in the domain of the function
     * \f$ f \f$. If this is not the case, the behaviour is undefined.
     *
     * It recursively performs \f$ 2^N - 1 \f$ 1-dimensional polynomial interpolations of degree \f$ m
     * \f$, where \f$ N \f$ corresponds to the number of dimensions.
     *
     * @param x Point \f$ \mathbf{x} \in \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(\mathbf{x}) \f$.
     */
    [[nodiscard]] double operator()(const grid_type::value_type& x) const {
        const auto idx_arr = index_subrange(grid_, npoints_, x);
        return interp_poly_nd<dim() - 1>(x, idx_arr);
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ \mathbf{x} = (x_1, \dots, x_N) \f$.
     *
     * @details The given point \f$ \mathbf{x} \f$ is assumed to lie in the domain of the function
     * \f$ f \f$. If this is not the case, the behaviour is undefined.
     *
     * It recursively performs \f$ 2^N - 1 \f$ 1-dimensional polynomial interpolations of degree \f$ m
     * \f$, where \f$ N \f$ corresponds to the number of dimensions.
     *
     * @tparam Vals Value types.
     * @param xs Values \f$ x_1, \dots, x_N \f$ of the point \f$ \mathbf{x} = (x_1, \dots, x_N) \in
     * \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(\mathbf{x}) \f$.
     */
    template <typename... Vals>
    [[nodiscard]] double operator()(Vals... xs) const {
        return this->operator()(typename grid_type::value_type { xs... });
    }

    /**
     * @brief Get the underlying grid \f$ g \f$ on which the function values \f$ f_{i_1, \cdots, i_N}
     * \f$ are evaluated.
     *
     * @return N-dimensional grid \f$ g \f$.
     */
    [[nodiscard]] const auto& grid() const { return grid_; }

    /**
     * @brief Get the function values \f$ f_{i_1, \cdots, i_N} = f(g(i_1, \dots, i_N)) \f$.
     *
     * @return `std::span` containing the function values \f$ f_{i_1, \cdots, i_N} \f$ at the grid
     * points \f$ g(i_1, \dots, i_N) \f$.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

    /**
     * @brief Get the degree \f$ m \f$.
     *
     * @return Degree of the interpolating polynomials \f$ h_i(x_i) \f$ in 1D.
     */
    [[nodiscard]] auto degree() const { return npoints_ - 1; }

private:
    // Polynomial interpolation given the point x and the index array of the bin at the lower left
    // corner of the volume in which to interpolate.
    //
    // This function is called recursively, with N being the current dimension.
    template <int N>
    double interp_poly_nd(const typename grid_type::value_type& x, const typename Grid::size_type& idx_arr) const {
        static_assert(N >= 0 && N < grid_type::dim(), "Invalid template parameter in interp_poly_nd.");
        auto idx_p_arr = idx_arr;
        const auto& grid_n = std::get<N>(grid_.grids());
        if constexpr (N == 0) {
            // perform simple polynomial interpolation in the first dimension
            for (std::size_t i = 0; i < npoints_; ++i) {
                idx_p_arr[N] = idx_arr[N] + i;
                fvecs_[N][i] = fvals_[flat_index(idx_p_arr, grid_.shape(), Order {})];
                xvecs_[N][i] = x[N] - grid_n.at(idx_p_arr[N]);
            }
            return detail::neville(fvecs_[N], xvecs_[N], npoints_);
        } else {
            // perform polynomial interpolation in N-1 dimensions
            for (std::size_t i = 0; i < npoints_; ++i) {
                idx_p_arr[N] = idx_arr[N] + i;
                fvecs_[N][i] = interp_poly_nd<N - 1>(x, idx_p_arr);
                xvecs_[N][i] = x[N] - grid_n.at(idx_p_arr[N]);
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
