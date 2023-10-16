/**
 * @file symmetric_power_grid.hpp
 * @brief 1-dimensional, symmetric power grid.
 */

#ifndef SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP
#define SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP

#include <simplemc/grids/power_grid.hpp>

namespace simplemc {

/**
 * @brief Lazy, 1-dimensional, symmetric power grid.
 *
 * @details The grid is defined by the value of the first grid point, the value of the
 * last grid point (!= first), its size (>= 2 && odd) and a power parameter (> 0).
 * Let c be the midpoint of the grid range, i.e. c = 0.5 * (first() + last()), and 
 * i_c = floor(size() / 2) + 1 be the corresponding index. Let g1 be the grid from first() 
 * to c with i_c points and let g2 be the descending grid from last() to c with i_c points 
 * such that g1(i_c) = g2(i_c). The grid points y(i) are calculated as follows:
 *
 *     y(i) = g1(i) if i <= i_c and
 *     y(i) = g2(size() - 1 - i) if i > i_c.
 */
class symmetric_power_grid : public grid_base {
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
     * @brief Constructor for the power_grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     * @param power Power parameter.
     */
    symmetric_power_grid(value_type first, value_type last, size_type size, value_type power);

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
        if (idx <= (size_ - 1) / 2) {
            return g1_.at(idx);
        } else {
            return g2_.at(size_ - 1 - idx);
        }   
    }

    /**
     * @brief Get index of bin to which the value belongs.
     *
     * @param value Input value.
     * @return Index of bin, which contains the supplied value.
     */
    [[nodiscard]] size_type index(value_type value) const override {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        if (value <= midpoint_) {
            return g1_.index(value);
        } else {
            return size_ - 1 - g2_.index(value);
        }
    }

    /**
     * @brief Get midpoint of grid.
     *
     * @return Grid value in the middle of the grid.
     */
    [[nodiscard]] value_type midpoint() const { return midpoint_; }

    /**
     * @brief Get power grid from first() to midpoint().
     *
     * @return Power grid.
     */
    [[nodiscard]] const auto& grid1() const { return g1_; }

    /**
     * @brief Get power grid from last() to midpoint().
     *
     * @return Power grid.
     */
    [[nodiscard]] const auto& grid2() const { return g2_; }

private:
    value_type midpoint_;
    power_grid g1_;
    power_grid g2_;
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP
