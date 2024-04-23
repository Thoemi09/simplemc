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
 * @addtogroup simplemc-numeric
 * @{
 */

namespace detail {

/**
 * @brief Get basic spline matrix, i.e. with the first and last equation left unspecified.
 * The remaining equations are specified by the boundary conditions.
 *
 * @details See LHS of Eq. 3.3.7 times 6 in Press, Numerical Recipes.
 *
 * @tparam Grid Type of grid in x direction.
 * @param grid Grid of x values.
 * @return Spline matrix.
 */
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

/**
 * @brief Get basic spline vector, i.e. with the first and last element left unspecified.
 * The remaining elements are specified by the boundary conditions.
 *
 * @details See RHS of Eq. 3.3.7 times 6 in Press, Numerical Recipes.
 *
 * @tparam Grid Type of grid in x direction.
 * @param grid Grid of x values.
 * @param fvals Function values at grid points.
 * @return Spline vector.
 */
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
 * @brief 1D cubic spline interpolation.
 *
 * @details The function values are not owned by the interpolation class. Only a span
 * to the original data is stored.
 *
 * @tparam Grid Type of grid in x direction.
 */
template <typename Grid>
class cubic_spline_interpolation {
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
     * @brief Construct a cubic spline interpolation object on a given grid with given function values.
     *
     * @details Natural cubic spline boundary conditions are used.
     *
     * @param grid Grid in x direction.
     * @param fvals Function values at grid points.
     */
    cubic_spline_interpolation(const grid_type& grid, const std::span<double>& fvals);

    /**
     * @brief Construct a cubic spline interpolation object on a given grid with given function values.
     *
     * @details The provided first two derivatives at the end points of the grid are
     * used as boundary conditions.
     *
     * @param grid Grid in x direction.
     * @param fvals Function values at grid points.
     * @param yp_0 First derivative at lower bound of grid.
     * @param yp_n First derivative at upper bound of grid.
     */
    cubic_spline_interpolation(const grid_type& grid, const std::span<double>& fvals, double yp_0, double yp_n);

    /**
     * @brief Construct a cubic spline interpolation object on a given grid with given function values.
     *
     * @details The user provided spline matrix and spline vector are used.
     *
     * @param grid Grid in x direction.
     * @param fvals Function values at grid points.
     * @param mat User specified spline matrix.
     * @param vec User specified spline vector.
     */
    cubic_spline_interpolation(
        const grid_type& grid, const std::span<double>& fvals, const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec);

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
    /**
     * @brief Set second derivatives by solving the linear system mat * y2 = vec.
     *
     * @param mat Spline matrix.
     * @param vec Spline vector.
     */
    void set_y2(const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) { y2_ = mat.colPivHouseholderQr().solve(vec); }

private:
    grid_type grid_;
    std::span<double> fvals_;
    Eigen::VectorXd y2_;
};

template <typename Grid>
cubic_spline_interpolation<Grid>::cubic_spline_interpolation(const grid_type& grid, const std::span<double>& fvals) :
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

template <typename Grid>
cubic_spline_interpolation<Grid>::cubic_spline_interpolation(
    const grid_type& grid, const std::span<double>& fvals, double yp_0, double yp_n) :
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
    vec(0) = yp_0 - (fvals_[1] - fvals_[0]) / (x1 - x0);
    const double xm1 = grid_.at(np - 1);
    const double xm2 = grid_.at(np - 2);
    mat(np - 1, np - 1) = 1.0 / 3.0 * (xm1 - xm2);
    mat(np - 1, np - 2) = 1.0 / 6.0 * (xm1 - xm2);
    vec(np - 1) = yp_n - (fvals_[np - 1] - fvals_[np - 2]) / (xm1 - xm2);
    set_y2(mat, vec);
}

template <typename Grid>
cubic_spline_interpolation<Grid>::cubic_spline_interpolation(
    const grid_type& grid, const std::span<double>& fvals, const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) :
    grid_(grid),
    fvals_(fvals),
    y2_(Eigen::VectorXd::Zero(grid_.size())) {
    if (grid_.size() != static_cast<grid_type::size_type>(fvals_.size())) {
        throw simplemc_exception("Number of grid points not equal to number of y values.",
            "cubic_spline_interpolation::cubic_spline_interpolation");
    }
    if (mat.rows() != mat.cols() || grid_.size() != mat.rows()) {
        throw simplemc_exception(
            "Size of provided spline matrix not correct.", "cubic_spline_interpolation::cubic_spline_interpolation");
    }
    if (grid_.size() != vec.size()) {
        throw simplemc_exception(
            "Size of provided spline vector not correct.", "cubic_spline_interpolation::cubic_spline_interpolation");
    }
    set_y2(mat, vec);
}

template <typename Grid>
double cubic_spline_interpolation<Grid>::operator()(double x) const {
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

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_CUBIC_SPLINE_INTERPOLATION_HPP
