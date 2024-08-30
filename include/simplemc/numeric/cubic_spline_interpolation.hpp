/**
 * @file
 * @brief Cubic spline interpolation in 1D.
 */

#ifndef SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <span>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-interpolation
 * @{
 */

namespace detail {

// Get basic spline matrix, i.e. with the first and last equation left unspecified, given the grid on
// the x-axis.
// See LHS of Eq. 3.3.7 times 6 in Press, Numerical Recipes.
template <typename Grid>
[[nodiscard]] Eigen::MatrixXd cubic_spline_matrix(const Grid& grid) {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(grid.size(), grid.size());
    for (long i = 1; i < grid.size() - 1; ++i) {
        auto xjm1 = grid.at(i - 1);
        auto xj = grid.at(i);
        auto xjp1 = grid.at(i + 1);
        mat(i, i - 1) = xj - xjm1;
        mat(i, i) = 2.0 * (xjp1 - xjm1);
        mat(i, i + 1) = xjp1 - xj;
    }
    return mat;
}

// Get basic spline vector, i.e. with the first and last element left unspecified, given the grid on
// the x-axis and the function values at the grid points.
// See RHS of Eq. 3.3.7 times 6 in Press, Numerical Recipes.
template <typename Grid>
[[nodiscard]] Eigen::VectorXd cubic_spline_vector(const Grid& grid, const std::span<double>& fvals) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(grid.size());
    for (long i = 1; i < grid.size() - 1; ++i) {
        auto xjm1 = grid.at(i - 1);
        auto xj = grid.at(i);
        auto xjp1 = grid.at(i + 1);
        vec(i) = 6.0 * ((fvals[i + 1] - fvals[i]) / (xjp1 - xj) - (fvals[i] - fvals[i - 1]) / (xj - xjm1));
    }
    return vec;
}

} // namespace detail

/**
 * @brief Cubic spline interpolation in 1D.
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
 * auto interp = simplemc::cubic_spline_interpolation(grid, fvals)
 *
 * // use the interpolation object ...
 * @endcode
 *
 * @tparam Grid 1-dimensional grid type.
 */
template <typename Grid>
class cubic_spline_interpolation {
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
     * @brief Construct a cubic spline interpolation object on a given grid with the given function
     * values.
     *
     * @details Natural cubic spline boundary conditions are used.
     *
     * @note See Eq. 3.3.7 times 6 in Press, Numerical Recipes.
     *
     * @param grid 1-dimensional grid on the \f$ x \f$ axis.
     * @param fvals Function values at the grid points.
     */
    cubic_spline_interpolation(const grid_type& grid, const std::span<double>& fvals) :
        grid_(grid),
        fvals_(fvals),
        y2_(Eigen::VectorXd::Zero(grid_.size())) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "cubic_spline_interpolation::cubic_spline_interpolation");
        }
        const auto np = grid_.size();
        auto mat = detail::cubic_spline_matrix(grid_);
        auto vec = detail::cubic_spline_vector(grid_, fvals_);
        mat(0, 0) = 1;
        vec(0) = 0;
        mat(np - 1, np - 1) = 1;
        vec(np - 1) = 0;
        set_y2(mat, vec);
    }

    /**
     * @brief Construct a cubic spline interpolation object on a given grid with the given function
     * values.
     *
     * @details The provided first two derivatives at the end points of the grid are used as boundary
     * conditions.
     *
     * @note See Eq. 3.3.7 times 6 in Press, Numerical Recipes.
     *
     * @param grid 1-dimensional grid on the \f$ x \f$ axis.
     * @param fvals Function values at the grid points.
     * @param fp_0 First derivative at lower bound of the grid.
     * @param fp_n First derivative at upper bound of the grid.
     */
    cubic_spline_interpolation(const grid_type& grid, const std::span<double>& fvals, double fp_0, double fp_n) :
        grid_(grid),
        fvals_(fvals),
        y2_(Eigen::VectorXd::Zero(grid_.size())) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "cubic_spline_interpolation::cubic_spline_interpolation");
        }
        const auto np = grid_.size();
        auto mat = detail::cubic_spline_matrix(grid_);
        auto vec = detail::cubic_spline_vector(grid_, fvals_);
        const double x0 = grid_.at(0);
        const double x1 = grid_.at(1);
        mat(0, 0) = -1.0 / 3.0 * (x1 - x0);
        mat(0, 1) = -1.0 / 6.0 * (x1 - x0);
        vec(0) = fp_0 - (fvals_[1] - fvals_[0]) / (x1 - x0);
        const double xm1 = grid_.at(np - 1);
        const double xm2 = grid_.at(np - 2);
        mat(np - 1, np - 1) = 1.0 / 3.0 * (xm1 - xm2);
        mat(np - 1, np - 2) = 1.0 / 6.0 * (xm1 - xm2);
        vec(np - 1) = fp_n - (fvals_[np - 1] - fvals_[np - 2]) / (xm1 - xm2);
        set_y2(mat, vec);
    }

    /**
     * @brief Construct a cubic spline interpolation object on a given grid with the given function
     * values.
     *
     * @details The user provided spline matrix and spline vector are used.
     *
     * @note See Eq. 3.3.7 times 6 in Press, Numerical Recipes.
     *
     * @param grid 1-dimensional grid on the \f$ x \f$ axis.
     * @param fvals Function values at the grid points.
     * @param mat User specified spline matrix.
     * @param vec User specified spline vector.
     */
    cubic_spline_interpolation(
        const grid_type& grid, const std::span<double>& fvals, const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) :
        grid_(grid),
        fvals_(fvals),
        y2_(Eigen::VectorXd::Zero(grid_.size())) {
        if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
            throw simplemc_exception("Number of grid points not equal to number of y values.",
                "cubic_spline_interpolation::cubic_spline_interpolation");
        }
        if (mat.rows() != mat.cols() || grid_.size() != mat.rows()) {
            throw simplemc_exception("Size of provided spline matrix not correct.",
                "cubic_spline_interpolation::cubic_spline_interpolation");
        }
        if (grid_.size() != vec.size()) {
            throw simplemc_exception("Size of provided spline vector not correct.",
                "cubic_spline_interpolation::cubic_spline_interpolation");
        }
        set_y2(mat, vec);
    }

    /**
     * @brief Perform the cubic spline interpolation.
     *
     * @param x Value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double operator()(double x) const {
        const auto idx = grid_.index_subrange(2, x);
        const double xp1 = grid_.at(idx + 1);
        const double xp = grid_.at(idx);
        const double h = xp1 - xp;
        const double a = (xp1 - x) / h;
        const double b = (x - xp) / h;
        const double y = a * fvals_[idx] + b * fvals_[idx + 1] +
            ((a * a * a - a) * y2_[idx] + (b * b * b - b) * y2_[idx + 1]) * h * h / 6.0;
        return y;
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
    // Set second derivatives by solving the linear system mat * y2 = vec, where mat is the spline
    // matrix and vec is the spline vector.
    void set_y2(const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) { y2_ = mat.colPivHouseholderQr().solve(vec); }

private:
    grid_type grid_;
    std::span<double> fvals_;
    Eigen::VectorXd y2_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP
