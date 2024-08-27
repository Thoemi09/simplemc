#include "../test_utils.hpp"

#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/nd_grid.hpp>

// Test 2-dimensional linear grid.
TEST(SimplemcGrids, TwoDimensionalLinearGrid) {
    simplemc::linear_grid lg { 0, 20, 11 };
    simplemc::nd_grid grid { lg, lg };
    using nd_size_type = decltype(grid)::nd_size_type;
    using nd_value_type = decltype(grid)::nd_value_type;
    ASSERT_EQ(grid.dim(), 2);
    ASSERT_EQ(grid.size(), lg.size() * lg.size());
    check_range_equal(grid.shape(), nd_size_type { lg.size(), lg.size() });
    check_range_near(grid.first(), nd_value_type { 0, 0 });
    check_range_near(grid.last(), nd_value_type { 20, 20 });
    auto view = grid.view();
    auto view_it = view.begin();
    auto view_center = grid.view_center();
    auto view_center_it = view_center.begin();
    auto view_bin_volumes = grid.view_bin_volumes();
    auto view_bin_volumes_it = view_bin_volumes.begin();
    for (int i = 0; i < lg.size(); ++i) {
        for (int j = 0; j < lg.size(); ++j) {
            nd_size_type idx_arr { i, j };
            nd_value_type val_arr { i * 2 + 0.1, j * 2 + 0.7 };
            check_range_near(grid.at(idx_arr), nd_value_type { i * 2.0, j * 2.0 });
            check_range_near(*(view_it++), nd_value_type { i * 2.0, j * 2.0 });
            if (i < lg.size() - 1 && j < lg.size() - 1) {
                check_range_near(grid.center(idx_arr), nd_value_type { i * 2.0 + 1.0, j * 2.0 + 1.0 });
                check_range_near(*(view_center_it++), nd_value_type { i * 2.0 + 1.0, j * 2.0 + 1.0 });
                check_range_equal(grid.index(val_arr), nd_size_type { i, j });
                ASSERT_DOUBLE_EQ(grid.bin_volume(idx_arr), lg.bin_volume(i) * lg.bin_volume(j));
                ASSERT_DOUBLE_EQ(*(view_bin_volumes_it++), lg.bin_volume(i) * lg.bin_volume(j));
            }
        }
    }
    check_range_equal(grid.index(1.2, 3.2), nd_size_type { 0, 1 });
    check_range_equal(grid.index(0.0, 0.0), nd_size_type { 0, 0 });
    check_range_equal(grid.index(5.0, 8.9999), nd_size_type { 2, 4 });
    check_range_equal(grid.index_subrange(5, 2.2, 8.2), nd_size_type { 0, 2 });
    check_range_equal(grid.index_subrange(2, 10.5, 11.5), nd_size_type { 5, 5 });
}
