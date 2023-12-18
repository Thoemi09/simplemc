/**
 * @file linear_grid.cpp
 * @brief 1-dimensional linear grid.
 */

#include <simplemc/grids/linear_grid.hpp>

namespace simplemc {

linear_grid::linear_grid(value_type first, value_type last, size_type size) {
    reset(first, last, size);
}

void linear_grid::reset(value_type first, value_type last, size_type size) {
    grid_base::reset(first, last, size);
    step_ = (last_ - first_) / (static_cast<double>(size_) - 1.0);
}

} // namespace simplemc