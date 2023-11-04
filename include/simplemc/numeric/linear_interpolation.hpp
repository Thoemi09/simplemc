/**
 * @file linear_interpolation.hpp
 * @brief Linear interpolation in 1D, 2D and 3D.
 */

#ifndef SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
#define SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP

#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>

#include <vector>

namespace simplemc {

/**
 * @brief Perform linear interpolation between two points f(x0) and f(x1).
 *
 * @param xd Ratio of distance in x-direction, i.e. (x - x0) / (x0 - x1).
 * @param f0 Function value at x0, i.e. f(x0).
 * @param f1 Function value at x1, i.e. f(x1).
 * @return Interpolated value.
 */
double interpolate_linear(double xd, double f0, double f1);

/**
 * @brief Perform bilinear interpolation on a square between four points f(x0, y0),
 * f(x1, y0), f(x0, y1) and f(x1, y1).
 *
 * @param xd Ratio of distance in x-direction, i.e. (x - x0) / (x0 - x1).
 * @param yd Ratio of distance in y-direction, i.e. (y - y0) / (y0 - y1).
 * @param f00 Function value at (x0, y0), i.e. f(x0, y0).
 * @param f10 Function value at (x1, y0), i.e. f(x1, y0).
 * @param f01 Function value at (x0, y1), i.e. f(x0, y1).
 * @param f11 Function value at (x1, y1), i.e. f(x1, y1).
 * @return Interpolated value.
 */
double interpolate_bilinear(double xd, double yd, double f00, double f10, double f01, double f11);

/**
 * @brief Perform trilinear interpolation on a cube between eight points f(x0, y0, z0),
 * f(x1, y0, z0), f(x0, y1, z0), f(x1, y1, z0), f(x0, y0, z1), f(x1, y0, z1), f(x0, y1,
 * z1) and f(x1, y1, z1).
 *
 * @param xd Ratio of distance in x-direction, i.e. (x - x0) / (x0 - x1).
 * @param yd Ratio of distance in y-direction, i.e. (y - y0) / (y0 - y1).
 * @param zd Ratio of distance in z-direction, i.e. (z - z0) / (z0 - z1).
 * @param f000 Function value at (x0, y0, z0), i.e. f(x0, y0, z0).
 * @param f100 Function value at (x1, y0, z0), i.e. f(x1, y0, z0).
 * @param f010 Function value at (x0, y1, z0), i.e. f(x0, y1, z0).
 * @param f110 Function value at (x1, y1, z0), i.e. f(x1, y1, z0).
 * @param f001 Function value at (x0, y0, z1), i.e. f(x0, y0, z1).
 * @param f101 Function value at (x1, y0, z1), i.e. f(x1, y0, z1).
 * @param f011 Function value at (x0, y1, z1), i.e. f(x0, y1, z1).
 * @param f111 Function value at (x1, y1, z1), i.e. f(x1, y1, z1).
 * @return Interpolated value.
 */
double interpolate_trilinear(double xd, double yd, double zd, double f000, double f100, double f010, double f110,
    double f001, double f101, double f011, double f111);

/**
 * @brief 1D linear interpolation.
 *
 * @details The function values and the grid are copied into the interpolation class. The XGrid
 * is required to implement the following methods/type aliases:
 *
 * - `XGrid::size_type`
 * - `size() -> std::integral`
 * - `index(double) -> std::integral`
 * - `at(std::integral) -> double`
 *
 * @tparam XGrid Type of grid in x direction.
 */
template <typename XGrid>
class linear_interpolation {
public:
    /**
     * @brief Grid type.
     */
    using xgrid_type = XGrid;

    /**
     * @brief Construct linear interpolation on a given grid with given function values.
     *
     * @tparam R Range type.
     * @param grid Grid in x direction.
     * @param fvals Function values at grid points.
     */
    template <ranges::range R>
    linear_interpolation(const xgrid_type& grid, R&& fvals);

    /**
     * @brief Perform interpolation.
     *
     * @param x Value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double interpolate(double x) const;

    /**
     * @brief Get grid in x direction.
     */
    [[nodiscard]] const auto& xgrid() const { return xgrid_; }

    /**
     * @brief Get function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    xgrid_type xgrid_;
    std::vector<double> fvals_;
};

template <typename XGrid>
template <ranges::range R>
linear_interpolation<XGrid>::linear_interpolation(const xgrid_type& grid, R&& fvals) : // NOLINT
    xgrid_(grid),
    fvals_(std::begin(fvals), std::end(fvals)) {
    if (grid.size() != static_cast<xgrid_type::size_type>(fvals_.size())) {
        throw simplemc_exception(
            "Number of grid points not equal to number of y values.", "linear_interpolation::linear_interplation");
    }
}

template <typename XGrid>
double linear_interpolation<XGrid>::interpolate(double x) const {
    const auto idx = index_of_subset(xgrid_.index(x), xgrid_.size(), static_cast<XGrid::size_type>(2));
    const auto xd = (x - xgrid_.at(idx)) / (xgrid_.at(idx + 1) - xgrid_.at(idx));
    return interpolate_linear(xd, fvals_[idx], fvals_[idx + 1]);
}

/**
 * @brief 2D bilinear interpolation.
 *
 * @details The function values and the grids are copied into the interpolation class.
 * The XGrid/YGrid is required to implement the following methods/type aliases:
 * - `XGrid::size_type`
 * - `size() -> std::integral`
 * - `index(double) -> std::integral`
 * - `at(std::integral) -> double`
 *
 * @tparam XGrid Type of grid in x direction.
 * @tparam YGrid Type of grid in y direction.
 */
template <typename XGrid, typename YGrid>
class bilinear_interpolation {
public:
    /**
     * @brief Grid type in x direction.
     */
    using xgrid_type = XGrid;

    /**
     * @brief Grid type in y direction.
     */
    using ygrid_type = YGrid;

    /**
     * @brief Construct bilinear interpolation on 2 given grids with given function values.
     * The function values are assumed to be in row-major order with the y-grid values as
     * the fastest index.
     *
     * @tparam R Range type.
     * @param xgrid Grid of x values.
     * @param ygrid Grid of y values.
     * @param fvals Function values at grid points.
     */
    template <ranges::range R>
    bilinear_interpolation(const xgrid_type& xgrid, const ygrid_type& ygrid, R&& fvals);

    /**
     * @brief Perform interpolation.
     *
     * @param x x-value at which we seek the function value.
     * @param y y-value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double interpolate(double x, double y) const;

    /**
     * @brief Get grid in x direction.
     */
    [[nodiscard]] const auto& xgrid() const { return xgrid_; }

    /**
     * @brief Get grid in y direction.
     */
    [[nodiscard]] const auto& ygrid() const { return ygrid_; }

    /**
     * @brief Get function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    xgrid_type xgrid_;
    ygrid_type ygrid_;
    std::vector<double> fvals_;
};

template <typename XGrid, typename YGrid>
template <ranges::range R>
bilinear_interpolation<XGrid, YGrid>::bilinear_interpolation(
    const xgrid_type& xgrid, const ygrid_type& ygrid, R&& fvals) : // NOLINT
    xgrid_(xgrid),
    ygrid_(ygrid),
    fvals_(std::begin(fvals), std::end(fvals)) {
    if (xgrid_.size() * ygrid_.size() != static_cast<xgrid_type::size_type>(fvals_.size())) {
        throw simplemc_exception(
            "Number of grid points not equal to number of y values.", "bilinear_interpolation::bilinear_interplation");
    }
}

template <typename XGrid, typename YGrid>
double bilinear_interpolation<XGrid, YGrid>::interpolate(double x, double y) const {
    const auto ix = index_of_subset(xgrid_.index(x), xgrid_.size(), static_cast<XGrid::size_type>(2));
    const auto iy = index_of_subset(ygrid_.index(y), ygrid_.size(), static_cast<YGrid::size_type>(2));
    const auto xd = (x - xgrid_.at(ix)) / (xgrid_.at(ix + 1) - xgrid_.at(ix));
    const auto yd = (y - ygrid_.at(iy)) / (ygrid_.at(iy + 1) - ygrid_.at(iy));
    const auto n = ygrid_.size();
    return interpolate_bilinear(
        xd, yd, fvals_[ix * n + iy], fvals_[(ix + 1) * n + iy], fvals_[ix * n + iy + 1], fvals_[(ix + 1) * n + iy + 1]);
}

/**
 * @brief 3D trilinear interpolation.
 *
 * @details The function values and the grids are copied into the interpolation class.
 * The XGrid/YGrid/ZGrid is required to implement the following methods/type aliases:
 * - `XGrid::size_type`
 * - `size() -> std::integral`
 * - `index(double) -> std::integral`
 * - `at(std::integral) -> double`
 *
 * @tparam XGrid Type of grid in x direction.
 * @tparam YGrid Type of grid in y direction.
 * @tparam ZGrid Type of grid in z direction.
 */
template <typename XGrid, typename YGrid, typename ZGrid>
class trilinear_interpolation {
public:
    /**
     * @brief Grid type in x direction.
     */
    using xgrid_type = XGrid;

    /**
     * @brief Grid type in y direction.
     */
    using ygrid_type = YGrid;

    /**
     * @brief Grid type in z direction.
     */
    using zgrid_type = ZGrid;

    /**
     * @brief Construct trilinear interpolation on 3 given grids with given function values.
     * The function values are assumed to be in row-major order with the z-grid values as
     * the fastest index.
     *
     * @tparam R Range type.
     * @param xgrid Grid of x values.
     * @param ygrid Grid of y values.
     * @param zgrid Grid of z values.
     * @param fvals Function values at grid points.
     */
    template <ranges::range R>
    trilinear_interpolation(const xgrid_type& xgrid, const ygrid_type& ygrid, const zgrid_type& zgrid, R&& fvals);

    /**
     * @brief Perform interpolation.
     *
     * @param x x-value at which we seek the function value.
     * @param y y-value at which we seek the function value.
     * @param z z-value at which we seek the function value.
     * @return Interpolated value.
     */
    [[nodiscard]] double interpolate(double x, double y, double z) const;

    /**
     * @brief Get grid in x direction.
     */
    [[nodiscard]] const auto& xgrid() const { return xgrid_; }

    /**
     * @brief Get grid in y direction.
     */
    [[nodiscard]] const auto& ygrid() const { return ygrid_; }

    /**
     * @brief Get grid in z direction.
     */
    [[nodiscard]] const auto& zgrid() const { return zgrid_; }

    /**
     * @brief Get function values.
     */
    [[nodiscard]] const auto& function_values() const { return fvals_; }

private:
    xgrid_type xgrid_;
    ygrid_type ygrid_;
    zgrid_type zgrid_;
    std::vector<double> fvals_;
};

template <typename XGrid, typename YGrid, typename ZGrid>
template <ranges::range R>
trilinear_interpolation<XGrid, YGrid, ZGrid>::trilinear_interpolation(
    const xgrid_type& xgrid, const ygrid_type& ygrid, const zgrid_type& zgrid, R&& fvals) : // NOLINT
    xgrid_(xgrid),
    ygrid_(ygrid),
    zgrid_(zgrid),
    fvals_(std::begin(fvals), std::end(fvals)) {
    if (xgrid_.size() * ygrid_.size() * zgrid_.size() != static_cast<xgrid_type::size_type>(fvals_.size())) {
        throw simplemc_exception("Number of grid points not equal to number of y values.",
            "trilinear_interpolation::trilinear_interplation");
    }
}

template <typename XGrid, typename YGrid, typename ZGrid>
double trilinear_interpolation<XGrid, YGrid, ZGrid>::interpolate(double x, double y, double z) const {
    const auto ix = index_of_subset(xgrid_.index(x), xgrid_.size(), static_cast<XGrid::size_type>(2));
    const auto iy = index_of_subset(ygrid_.index(y), ygrid_.size(), static_cast<XGrid::size_type>(2));
    const auto iz = index_of_subset(zgrid_.index(z), zgrid_.size(), static_cast<XGrid::size_type>(2));
    const auto xd = (x - xgrid_.at(ix)) / (xgrid_.at(ix + 1) - xgrid_.at(ix));
    const auto yd = (y - ygrid_.at(iy)) / (ygrid_.at(iy + 1) - ygrid_.at(iy));
    const auto zd = (z - zgrid_.at(iz)) / (zgrid_.at(iz + 1) - zgrid_.at(iz));
    const auto ny = ygrid_.size();
    const auto nz = zgrid_.size();
    return interpolate_trilinear(xd, yd, zd, fvals_[(ix * ny + iy) * nz + iz], fvals_[((ix + 1) * ny + iy) * nz + iz],
        fvals_[(ix * ny + (iy + 1)) * nz + iz], fvals_[((ix + 1) * ny + (iy + 1)) * nz + iz],
        fvals_[(ix * ny + iy) * nz + iz + 1], fvals_[((ix + 1) * ny + iy) * nz + iz + 1],
        fvals_[(ix * ny + (iy + 1)) * nz + iz + 1], fvals_[((ix + 1) * ny + (iy + 1)) * nz + iz + 1]);
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LINEAR_INTERPOLATION_HPP
