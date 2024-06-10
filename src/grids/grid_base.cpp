/**
 * @file
 * @brief Implementation details for simplemc/grids/grid_base.hpp.
 */

#include <simplemc/grids/grid_base.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

grid_base::grid_base(value_type first, value_type last, size_type size) {
    reset(first, last, size);
}

void grid_base::reset(value_type first, value_type last, size_type size) {
    if (size < 2) {
        throw simplemc_exception("Grid size must be >= 2", "grid_base::reset");
    }
    if (first == last) {
        throw simplemc_exception("First and last grid value have to be different", "grid_base::reset");
    }
    first_ = first;
    last_ = last;
    size_ = size;
}

} // namespace simplemc
