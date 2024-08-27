/**
 * @file
 * @brief 1-dimensional symmetric power grid.
 */

#ifndef SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP
#define SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP

#include <simplemc/grids/power_grid.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-grids-1d
 * @brief 1-dimensional symmetric power grid.
 *
 * @details This class inherits from simplemc::grid_base and represents a symmetric power grid of odd
 * size \f$ M \geq 2 \f$ on the interval \f$ [a, b] \f$ (or \f$ [b, a] \f$ if \f$ b < a \f$).
 *
 * It uses the map
 * \f[
 *   g(i) =
 *   \begin{cases}
 *   g_1(i) & \text{if } i \leq i_c, \\
 *   g_2(M - 1 - i) & \text{if } i > i_c
 *   \end{cases}
 *   \;,
 * \f]
 * where \f$ c = \frac{a + b}{2} \f$ is the midpoint of the grid and \f$ i_c = \left\lfloor
 * \frac{M}{2} \right\rfloor \f$ is the corresponding index. \f$ g_1 \f$ (\f$ g_2 \f$) is a
 * simplemc::power_grid of size \f$ \left\lfloor \frac{M}{2} \right\rfloor + 1 \f$ with the first grid
 * point at \f$ a \f$ (\f$ b \f$) and the last grid point at \f$ c \f$.
 *
 * The inverse map is then given by
 * \f[
 *   g^{-1}(x) =
 *   \begin{cases}
 *   g_1^{-1}(x) & \text{if } x \leq c, \\
 *   M - 1 - g_2^{-1}(x) & \text{if } x > c
 *   \end{cases}
 *   \;.
 * \f]
 */
class symmetric_power_grid : public grid_base {
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
    symmetric_power_grid(value_type first, value_type last, size_type size, value_type power) :
        grid_base(first, last, size),
        midpoint_((first_ + last_) / 2),
        g1_(first_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power),
        g2_(last_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power) {
        if (size_ % 2 != 1) {
            throw simplemc_exception("Number of grid points needs to be odd in symmetric_power_grid",
                "symmetric_power_grid::check_odd_size");
        }
    }

    /**
     * @brief Reset a power grid by specifying its first and last grid points, its size and the power
     * parameter.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     * @param power Power parameter.
     */
    void reset(value_type first, value_type last, size_type size, value_type power) {
        if (size % 2 != 1) {
            throw simplemc_exception("Number of grid points needs to be odd in symmetric_power_grid",
                "symmetric_power_grid::check_odd_size");
        }
        if (power <= 0) {
            throw simplemc_exception("Power parameter must be > 0", "symmetric_power_grid::reset");
        }
        grid_base::reset(first, last, size);
        midpoint_ = (first_ + last_) / 2;
        g1_.reset(first_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power);
        g2_.reset(last_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power);
    }

    /**
     * @brief Get the grid point at a given index.
     *
     * @param idx Index of the grid point.
     * @return Grid point at the given index.
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
     * @brief Get the index of the bin to which a given value belongs.
     *
     * @param value A value in the range of the grid.
     * @return Index of the bin which contains the given value.
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
     * @brief Get the midpoint of the grid.
     *
     * @return Grid point in the middle of the grid.
     */
    [[nodiscard]] value_type midpoint() const { return midpoint_; }

    /**
     * @brief Get the power grid from first() to midpoint().
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
