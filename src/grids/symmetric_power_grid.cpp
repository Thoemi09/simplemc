/**
 * @file
 * @brief 1-dimensional, symmetric power grid.
 */

#include <simplemc/grids/symmetric_power_grid.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

symmetric_power_grid::symmetric_power_grid(value_type first, value_type last, size_type size, value_type power) :
    grid_base(first, last, size),
    midpoint_((first_ + last_) / 2),
    g1_(first_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power),
    g2_(last_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power) {
    if (size_ % 2 != 1) {
        throw simplemc_exception(
            "Number of grid points needs to be odd in symmetric_power_grid", "symmetric_power_grid::check_odd_size");
    }
}

void symmetric_power_grid::reset(value_type first, value_type last, size_type size, value_type power) {
    if (size % 2 != 1) {
        throw simplemc_exception(
            "Number of grid points needs to be odd in symmetric_power_grid", "symmetric_power_grid::check_odd_size");
    }
    if (power <= 0) {
        throw simplemc_exception("Power parameter must be > 0", "symmetric_power_grid::reset");
    }
    grid_base::reset(first, last, size);
    midpoint_ = (first_ + last_) / 2;
    g1_.reset(first_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power);
    g2_.reset(last_, midpoint_, static_cast<size_type>(size_ / 2) + 1, power);
}

} // namespace simplemc