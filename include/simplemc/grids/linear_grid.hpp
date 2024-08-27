/**
 * @file
 * @brief 1-dimensional linear grid.
 */

#ifndef SIMPLEMC_GRIDS_LINEAR_GRID_HPP
#define SIMPLEMC_GRIDS_LINEAR_GRID_HPP

#include <simplemc/grids/grid_base.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-grids-1d
 * @brief 1-dimensional linear grid.
 *
 * @details This class inherits from simplemc::grid_base and represents a uniform grid of size
 * \f$ M \geq 2 \f$ on the interval \f$ [a, b] \f$ (or \f$ [b, a] \f$ if \f$ b < a \f$).
 *
 * It uses the map
 * \f[
 *   g(i) = a + i * \Delta \;,
 * \f]
 * with the step size \f$ \Delta = \frac{b - a}{M - 1} \f$ and the corresponding inverse map
 * \f[
 *   g^{-1}(x) = \left\lfloor \frac{x - a}{\Delta} \right\rfloor \;,
 * \f]
 * with \f$ x \in [a, b] \f$.
 */
class linear_grid : public grid_base {
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
     * @brief Construct a linear grid by specifying its first and last grid points and its size.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    linear_grid(value_type first, value_type last, size_type size) { reset(first, last, size); }

    /**
     * @brief Reset the linear grid by specifying its first and last grid points and its size.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    void reset(value_type first, value_type last, size_type size) {
        grid_base::reset(first, last, size);
        step_ = (last_ - first_) / (static_cast<double>(size_) - 1.0);
    }

    /**
     * @brief Get the grid point at a given index.
     *
     * @param idx Index of the grid point.
     * @return Grid point at the given index.
     */
    [[nodiscard]] value_type at(size_type idx) const override {
        assert(idx >= 0 && idx < size_);
        return first_ + step_ * static_cast<double>(idx);
    }

    /**
     * @brief Get the index of the bin to which a given value belongs.
     *
     * @param value A value in the range of the grid.
     * @return Index of the bin which contains the given value.
     */
    [[nodiscard]] size_type index(value_type value) const override {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        return static_cast<size_type>((value - first_) / step_);
    }

    /**
     * @brief Get the volume (size) of the bin at a given index.
     *
     * @param idx Bin index.
     * @return Bin volume (size) at the given index.
     */
    [[nodiscard]] value_type bin_volume([[maybe_unused]] size_type idx) const override {
        assert(idx >= 0 && idx + 1 < size_);
        return std::abs(step_);
    }

    /**
     * @brief Get the center of the bin at a given index.
     *
     * @param idx Bin index.
     * @return Center of the bin at the given index.
     */
    [[nodiscard]] value_type center(size_type idx) const override {
        assert(idx >= 0 && idx + 1 < size_);
        return at(idx) + step_ / 2.;
    }

    /**
     * @brief Get the step size between two adjacent grid points.
     *
     * @return Distance between two adjacent grid points.
     */
    [[nodiscard]] value_type step() const { return step_; }

private:
    value_type step_ { 0.0 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_LINEAR_GRID_HPP
