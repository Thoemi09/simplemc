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
 * @details The grid is defined by the value of the first grid point, the value of the
 * last grid point (!= first), its size (>= 2) and a power parameter (> 0). The grid points
 * `y(i)` are generated according to `y(i) = first + scale * i**power`, where `i in [0, 1, ..., size - 1]`
 * and `scale = (last - first) / std::pow((size - 1), power)`.
 *
 * If last < first, then the grid is descending.
 */
class power_grid : public grid_base {
public:
    /**
     * @brief Value type of the grid.
     */
    using value_type = grid_base::value_type;

    /**
     * @brief Size type of the grid.
     */
    using size_type = grid_base::size_type;

    /**
     * @brief Constructor for a power grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     * @param power Power parameter.
     */
    power_grid(value_type first, value_type last, size_type size, value_type power);

    /**
     * @brief Reset the grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     * @param power Power parameter.
     */
    void reset(value_type first, value_type last, size_type size, value_type power);

    /**
     * @brief Get grid point at a certain index.
     *
     * @param idx Index of grid point.
     * @return Value at that index.
     */
    [[nodiscard]] value_type at(size_type idx) const override {
        assert(idx >= 0 && idx < size_);
        return first_ + scale_ * std::pow(idx, power_);
    }

    /**
     * @brief Get index of bin to which the value belongs.
     *
     * @param value Input value.
     * @return Index of bin, which contains the supplied value.
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
     * @brief Get the scale parameter of the grid.
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
