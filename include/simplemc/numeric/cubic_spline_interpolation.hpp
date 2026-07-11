// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Cubic spline interpolation in 1D.
 */

#ifndef SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP

#include <simplemc/grids/concepts.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
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

// Get the basic spline matrix for a given grid (first and last equation/row are left unspecified).
// See LHS of Eq. 3.3.7 times 6 in Press, Numerical Recipes.
template <typename Grid>
[[nodiscard]] Eigen::MatrixXd cubic_spline_matrix(const Grid& grid) {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(grid.size(), grid.size());
    for (long i = 1; i < grid.size() - 1; ++i) {
        const auto xim1 = grid.at(i - 1);
        const auto xi = grid.at(i);
        const auto xip1 = grid.at(i + 1);
        mat(i, i - 1) = xi - xim1;
        mat(i, i) = 2.0 * (xip1 - xim1);
        mat(i, i + 1) = xip1 - xi;
    }
    return mat;
}

// Get basic spline vector for a given the grid and given function values at the grid points (first
// and last element are left unspecified).
// See RHS of Eq. 3.3.7 times 6 in Press, Numerical Recipes.
template <typename Grid>
[[nodiscard]] Eigen::VectorXd cubic_spline_vector(const Grid& grid, const std::span<double>& fvals) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(grid.size());
    for (long i = 1; i < grid.size() - 1; ++i) {
        const auto xim1 = grid.at(i - 1);
        const auto xi = grid.at(i);
        const auto xip1 = grid.at(i + 1);
        vec(i) = 6.0 * ((fvals[i + 1] - fvals[i]) / (xip1 - xi) - (fvals[i] - fvals[i - 1]) / (xi - xim1));
    }
    return vec;
}

} // namespace detail

/**
 * @brief Cubic spline interpolation in 1D.
 *
 * @details The cubic spline interpolant \f$ h(x) \f$ of the function \f$ f(x) \f$ is defined as the
 * piecewise polynomial
 * \f[
 *   h(x) =
 *   \begin{cases}
 *     h_0(x) &= a_0 + b_0 (x - x_0) + c_0 (x - x_0)^2 + d_0 (x - x_0)^3 & \text{if } x \in [x_0, x_1]
 *     \\
 *     h_1(x) &= a_1 + b_1 (x - x_1) + c_1 (x - x_1)^2 + d_1 (x - x_1)^3 & \text{if } x \in [x_1, x_2]
 *     \\ \vdots \\
 *     h_{M-2}(x) &= a_{M-2} + b_{M-2} (x - x_{M-2}) + c_{M-2} (x - x_{M-2})^2 + d_{M-2} (x -
 *     x_{M-2})^3 & \text{if } x \in [x_{M-2}, x_{M-1}] \\
 *   \end{cases}
 *   \; ,
 * \f]
 * where the functions \f$ h_i(x) \f$ are required to satisfy
 * - \f$ h_i(x_i) = f_i \f$ and \f$ h_i(x_{i+1}) = f_{i+1} \f$,
 * - \f$ h'_i(x_{i+1}) = h'_{i+1}(x_{i+1}) \f$ for \f$ 0 \leq i < M - 2 \f$ and
 * - \f$ h''_i(x_{i+1}) = h''_{i+1}(x_{i+1}) \f$ for \f$ 0 \leq i < M - 2 \f$.
 *
 * Here, \f$ f_i = f(x_i) \f$ is the function value at the point \f$ x_i = g(i) \f$.
 *
 * The requirements give us \f$ 4(M - 1) - 2 \f$ equations for the \f$ 4(M-1) \f$ unknown coefficients
 * \f$ a_i \f$, \f$ b_i \f$, \f$ c_i \f$ and \f$ d_i \f$. The remaining 2 boundary conditions can be
 * specified by the user.
 *
 * Under the hood, the equations are rewritten in terms of the second derivatives \f$ h''_i(x_i) = 2
 * c_i = \gamma_i \f$, which reduces their number to \f$ M - 2 \f$:
 * \f[
 *   \gamma_{i-1} (x_i - x_{i-1}) + 2 \gamma_i (x_{i+1} - x_{i-1}) + \gamma_{i+1} (x_{i+1} - x_i) =
 *   6 \left( \frac{f_{i+1} - f_i}{x_{i+1} - x_i} - \frac{f_{i} - f_{i-1}}{x_i - x_{i-1}} \right) \; ,
 *   \quad \forall i = 1, \dots, M - 2 \; .
 * \f]
 * Once the additional 2 equations are specified, we can solve the system \f$ \mathbf{A}
 * \mathbf{\gamma} = \mathbf{b} \f$ for the unknown second derivates \f$ \mathbf{\gamma} = (\gamma_0,
 * \dots, \gamma_{M-1}) \f$ and use
 * \f[
 *   h_i(x) = f_i + \left( \frac{f_{i+1} - f_i}{x_{i+1} - x_i} - \frac{x_{i+1} - x_i}{6}(2 \gamma_i +
 *   \gamma_{i+1}) \right) (x - x_i) + \frac{\gamma_i}{2} (x - x_i)^2 + \frac{\gamma_{i+1} - \gamma_i}
 *   {6 (x_{i+1} - x_i)}(x - x_i)^3 \; ,
 * \f]
 * to evaluate the interpolating polynomials.
 *
 * The class stores
 * - a simplemc::grid_1d \f$ g \f$ (see @ref simplemc-grids-1d) that determines the grid points \f$
 * g(i) = x_i \f$,
 * - the function values \f$ f_i = f(g(i)) \f$ at the grid points \f$ g(i) \f$ and
 * - the calculated second derivatives \f$ h''_i(x_i) = \gamma_i \f$.
 *
 * @note The function values are not owned by the interpolation class. Only a `std::span` to the
 * original data is stored.
 *
 * @include numeric/doc_cubic_splines.cpp
 *
 * Output:
 *
 * ```
 * x         interp(x)           f(x)
 * 0.0       0                   0
 * 0.2       0.0015968829        0.0016
 * 0.4       0.025598756         0.0256
 * 0.6       0.12959877          0.1296
 * 0.8       0.40959722          0.4096
 * 1.0       1                   1
 * 1.2       2.0735971           2.0736
 * 1.4       3.8415975           3.8416
 * 1.6       6.5536316           6.5536
 * 1.8       10.498769           10.4976
 * 2.0       16                  16
 * ```
 *
 * @tparam Grid simplemc::grid_1d grid type.
 */
template <grid_1d Grid>
class cubic_spline_interpolation {
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
     * @brief Construct a cubic spline interpolation object on a given grid \f$ g \f$ with the given
     * function values \f$ f_i = f(x_i) = f(g(i)) \f$ and natural boundary conditions.
     *
     * @details Natural boundary conditions set the second derivatives at the end points to zero, i.e.
     * \f$ \gamma_0 = \gamma_{M-1} = 0 \f$. This adds the following two missing equations to the
     * linear system:
     * \f[
     *   \gamma_0 = 0 \quad \text{and} \quad \gamma_{M-1} = 0 \; .
     * \f]
     *
     * It throws a simplemc::simplemc_exception if the size of the grid \f$ M \f$ is not equal to the
     * number of function values.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g 1-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_i \f$.
     */
    cubic_spline_interpolation(grid_type g, const std::span<double>& fvals) :
        grid_(std::move(g)),
        fvals_(fvals),
        gamma_(Eigen::VectorXd::Zero(grid_.size())) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values");
        }
        const auto np = grid_.size();
        auto mat = detail::cubic_spline_matrix(grid_);
        auto vec = detail::cubic_spline_vector(grid_, fvals_);
        mat(0, 0) = 1;
        vec(0) = 0;
        mat(np - 1, np - 1) = 1;
        vec(np - 1) = 0;
        set_gamma(mat, vec);
    }

    /**
     * @brief Construct a cubic spline interpolation object on a given grid \f$ g \f$ with the given
     * function values \f$ f_i = f(x_i) = f(g(i)) \f$ and fixed first derivatives at the end points.
     *
     * @details The provided first two derivatives, \f$ h'_0(x_0) = b_0 \f$ and \f$ h'_{M-2}(x_{M-1})
     * = b_{M-1} \f$ at the end points of the grid are used as boundary conditions. This adds the
     * following two missing equations to the linear system:
     * \f{eqnarray}{
     *   2 \gamma_0 (x_1 - x_0) + \gamma_1 (x_1 - x_0)  &=& 6 \left( \frac{f_1 - f_0}{x_1 - x_0} - b_0
     *   \right) \; , \\
     *   \gamma_{M-2} (x_{M-1} - x_{M-2}) + 2 \gamma_{M-1} (x_{M-1} - x_{M-2}) &=& 6 \left( b_{M-1} -
     *   \frac{f_{M-1} - f_{M-2}}{x_{M-1} - x_{M-2}} \right) \; .
     * \f}
     *
     * It throws a simplemc::simplemc_exception if the size of the grid \f$ M \f$ is not equal to the
     * number of function values.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g 1-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_i \f$.
     * @param b_0 Value of the first derivative at \f$ x_0 \f$, i.e. \f$ h'_0(x_0) = b_0 \f$.
     * @param b_m1  Value of the first derivative at \f$ x_{M-1} \f$, i.e. \f$ h'_{M-2}(x_{M-1}) =
     * b_{M-1} \f$.
     */
    cubic_spline_interpolation(grid_type g, const std::span<double>& fvals, double b_0, double b_m1) :
        grid_(std::move(g)),
        fvals_(fvals),
        gamma_(Eigen::VectorXd::Zero(grid_.size())) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values");
        }
        const auto np = grid_.size();
        auto mat = detail::cubic_spline_matrix(grid_);
        auto vec = detail::cubic_spline_vector(grid_, fvals_);
        const double x0 = grid_.at(0);
        const double x1 = grid_.at(1);
        mat(0, 0) = 2.0 * (x1 - x0);
        mat(0, 1) = x1 - x0;
        vec(0) = 6 * ((fvals_[1] - fvals_[0]) / (x1 - x0) - b_0);
        const double xm1 = grid_.at(np - 1);
        const double xm2 = grid_.at(np - 2);
        mat(np - 1, np - 1) = 2.0 * (xm1 - xm2);
        mat(np - 1, np - 2) = xm1 - xm2;
        vec(np - 1) = 6 * (b_m1 - (fvals_[np - 1] - fvals_[np - 2]) / (xm1 - xm2));
        set_gamma(mat, vec);
    }

    /**
     * @brief Construct a cubic spline interpolation object on a given grid \f$ g \f$ with the given
     * function values \f$ f_i = f(x_i) = f(g(i)) \f$ and the spline matrix and spline vector.
     *
     * @details The spline matrix \f$ \mathbf{A} \f$ and spline vector \f$ \mathbf{b} \f$ determine
     * the second derivatives \f$ \mathbf{\gamma} \f$ of the cubic spline interpolant via the linear
     * system \f$ \mathbf{A} \mathbf{\gamma} = \mathbf{b} \f$.
     *
     * It throws a simplemc::simplemc_exception if
     * - the size of the grid \f$ M \f$ is not equal to the number of function values,
     * - the spline matrix is not of size \f$ M \times M \f$ or
     * - the spline vector is not of size \f$ M \f$.
     *
     * @note The function values are not owned by the interpolation object. Only a `std::span` to the
     * original data is stored.
     *
     * @param g 1-dimensional grid \f$ g \f$.
     * @param fvals Function values \f$ f_i \f$.
     * @param a_mat Spline matrix \f$ \mathbf{A} \f$.
     * @param b_vec Spline vector \f$ \mathbf{b} \f$.
     */
    cubic_spline_interpolation(
        grid_type g, const std::span<double>& fvals, const Eigen::MatrixXd& a_mat, const Eigen::VectorXd& b_vec) :
        grid_(std::move(g)),
        fvals_(fvals),
        gamma_(Eigen::VectorXd::Zero(grid_.size())) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values");
        }
        if (a_mat.rows() != a_mat.cols() || grid_.size() != a_mat.rows()) {
            throw simplemc_exception("Size of provided spline matrix not correct");
        }
        if (grid_.size() != b_vec.size()) {
            throw simplemc_exception("Size of provided spline vector not correct");
        }
        set_gamma(a_mat, b_vec);
    }

    /**
     * @brief Evaluate the interpolant \f$ h \f$ at the point \f$ x \f$.
     *
     * @details The given point \f$ x \f$ is assumed to lie in the domain of the function \f$ f \f$.
     * If this is not the case, the behaviour is undefined.
     *
     * The value of the interpolant at \f$ x \f$ is given by
     * \f[
     *   h_i(x) = f_i + \left( \frac{f_{i+1} - f_i}{x_{i+1} - x_i} - \frac{x_{i+1} - x_i}{6}(2
     *   \gamma_i + \gamma_{i+1}) \right) (x - x_i) + \frac{\gamma_i}{2} (x - x_i)^2 +
     *   \frac{\gamma_{i+1} - \gamma_i} {6 (x_{i+1} - x_i)}(x - x_i)^3 \; ,
     * \f]
     * where \f$ x \in [x_i, x_{i+1}] \f$.
     *
     * @param x Point \f$ x \in \mathrm{D} \f$ at which to interpolate.
     * @return Interpolated value \f$ h(x) \f$.
     */
    [[nodiscard]] double operator()(double x) const {
        // here we follow Press, Numerical Recipes, although the equation above works as well
        const auto idx = index_subrange(grid_, 2, x);
        const double xp1 = grid_.at(idx + 1);
        const double xp = grid_.at(idx);
        const double h = xp1 - xp;
        const double a = (xp1 - x) / h;
        const double b = (x - xp) / h;
        const double y = a * fvals_[idx] + b * fvals_[idx + 1] +
            ((a * a * a - a) * gamma_[idx] + (b * b * b - b) * gamma_[idx + 1]) * h * h / 6.0;
        return y;
    }

    /**
     * @brief Get the underlying grid \f$ g \f$ on which the function values \f$ f_i \f$ are evaluated.
     *
     * @return 1-dimensional grid \f$ g \f$.
     */
    [[nodiscard]] const auto& grid() const noexcept { return grid_; }

    /**
     * @brief Get the function values \f$ f_i = f(g(i)) \f$.
     *
     * @return `std::span` containing the function values \f$ f_i \f$ at the grid points \f$ g(i) \f$.
     */
    [[nodiscard]] const auto& function_values() const noexcept { return fvals_; }

private:
    // Set second derivatives by solving the linear system A * \gamma = b, where A is the spline
    // matrix and b is the spline vector.
    void set_gamma(const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) {
        gamma_ = mat.colPivHouseholderQr().solve(vec);
    }

private:
    grid_type grid_;
    std::span<double> fvals_;
    Eigen::VectorXd gamma_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP
