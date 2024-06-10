/**
 * @file
 * @brief 1-dimensional power grid.
 */

#ifndef SIMPLEMC_GRIDS_POWER_GRID_HPP
#define SIMPLEMC_GRIDS_POWER_GRID_HPP

#include <simplemc/grids/grid_base.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-grids
 * @brief Lazy, 1-dimensional power grid.
 *
 * @details This class inherits from simplemc::grid_base and represents a power grid of size
 * \f$ N \geq 2 \f$ on the interval \f$ [a, b] \f$ (or \f$ [b, a] \f$ if \f$ b < a \f$).
 *
 * It uses the map
 * \f[
 *   g(i) = a + i^p * \sigma \;,
 * \f]
 * with the power parameter \f$ p > 0 \f$ and the scale factor \f$ \sigma = \frac{b - a}{(N - 1)^p}
 * \f$.
 *
 * The inverse map is given by
 * \f[
 *   g^{-1}(x) = \left\lfloor \left( \frac{x - a}{\sigma} \right)^{1/p} \right\rfloor \;,
 * \f]
 * with \f$ x \in [a, b] \f$.
 */
class power_grid : public grid_base {
public:
    /**
     * @brief Type of the grid's range, i.e. the type of the grid points.
     */
    using value_type = grid_base::value_type;

    /**
     * @brief Type of the grid's domain, i.e. the type of the grid indices.
     */
    using size_type = grid_base::size_type;

    /**
     * @brief Construct a power grid by specifying its first and last grid points, its size and the
     * power parameter.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     * @param power Power parameter.
     */
    power_grid(value_type first, value_type last, size_type size, value_type power);

    /**
     * @brief Reset a power grid by specifying its first and last grid points, its size and the power
     * parameter.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     * @param power Power parameter.
     */
    void reset(value_type first, value_type last, size_type size, value_type power);

    /**
     * @brief Get the grid point at a given index.
     *
     * @param idx Index of the grid point.
     * @return Grid point at the given index.
     */
    [[nodiscard]] value_type at(size_type idx) const override {
        assert(idx >= 0 && idx < size_);
        return first_ + scale_ * std::pow(idx, power_);
    }

    /**
     * @brief Get the index of the bin to which a given value belongs.
     *
     * @param value A value in the range of the grid.
     * @return Index of the bin which contains the given value.
     */
    [[nodiscard]] size_type index(value_type value) const override {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        return static_cast<size_type>(std::pow((value - first_) / scale_, 1.0 / power_));
    }

    /**
     * @brief Get the power parameter of the grid.
     *
     * @return Power parameter.
     */
    [[nodiscard]] value_type power() const { return power_; }

    /**
     * @brief Get the scale factor of the grid.
     *
     * @return Scaling of the distance between two grid points.
     */
    [[nodiscard]] value_type scale() const { return scale_; }

private:
    value_type power_ { 0.0 };
    value_type scale_ { 0.0 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_POWER_GRID_HPP
