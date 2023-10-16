/**
 * @file power_grid.cpp
 * @brief 1-dimensional power grid.
 */

#include <simplemc/grids/power_grid.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

power_grid::power_grid(value_type first, value_type last, size_type size, value_type power) {
    reset(first, last, size, power);
}

void power_grid::reset(value_type first, value_type last, size_type size, value_type power) {
    if (power <= 0) {
        throw simplemc_exception("Power parameter must be > 0.", "power_grid::reset");
    }
    grid_base::reset(first, last, size);
    power_ = power;
    scale_ = (last_ - first_) / std::pow(size_ - 1, power_);
}

} // namespace simplemc