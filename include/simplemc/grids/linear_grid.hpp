/**
 * @file
 * @brief 1-dimensional linear grid.
 */

#ifndef SIMPLEMC_GRIDS_LINEAR_GRID_HPP
#define SIMPLEMC_GRIDS_LINEAR_GRID_HPP

#include <simplemc/grids/grid_base.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-grids
 * @brief Lazy, 1-dimensional linear grid.
 *
 * @details The grid is defined by the value of the first grid point, the value of the
 * last grid point (!= first) and its size (>= 2). The grid points `y(i)` are generated
 * according to `y(i) = first + step * i`, where `i in [0, 1, ..., size - 1]` and
 * `step = (last - first) / (size - 1)`.
 *
 * If last < first, then the grid is descending.
 */
class linear_grid : public grid_base {
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
     * @brief Constructor for a linear grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    linear_grid(value_type first, value_type last, size_type size);

    /**
     * @brief Reset the grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    void reset(value_type first, value_type last, size_type size);

    /**
     * @brief Get grid point at a certain index.
     *
     * @param idx Index of grid point.
     * @return Value at that index.
     */
    [[nodiscard]] value_type at(size_type idx) const override {
        assert(idx >= 0 && idx < size_);
        return first_ + step_ * static_cast<double>(idx);
    }

    /**
     * @brief Get index of bin to which the value belongs.
     *
     * @param value Input value.
     * @return Index of bin, which contains the supplied value.
     */
    [[nodiscard]] size_type index(value_type value) const override {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        return static_cast<size_type>((value - first_) / step_);
    }

    /**
     * @brief Get volume of the bin at a certain index.
     *
     * @param idx Index of bin.
     * @return Bin volume at that index.
     */
    [[nodiscard]] value_type bin_volume([[maybe_unused]] size_type idx) const override {
        assert(idx >= 0 && idx + 1 < size_);
        return std::abs(step_);
    }

    /**
     * @brief Get the center of the bin at a certain index.
     *
     * @param idx Index of bin.
     * @return Center of bin at that index.
     */
    [[nodiscard]] value_type center(size_type idx) const override {
        assert(idx >= 0 && idx + 1 < size_);
        return at(idx) + step_ / 2.;
    }

    /**
     * @brief Get step size.
     *
     * @return Distance between two adjacent grid points.
     */
    [[nodiscard]] value_type step() const { return step_; }

private:
    value_type step_ { 0.0 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_LINEAR_GRID_HPP
