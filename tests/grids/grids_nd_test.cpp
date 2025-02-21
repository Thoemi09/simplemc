#include "../test_utils.hpp"

#include <range/v3/all.hpp>
#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/nd_grid.hpp>
#include <simplemc/grids/power_grid.hpp>

#include <vector>

// Test 2-dimensional default constructed linear - power grid.
TEST(SimplemcGrids, TwoDimensionalLinearPowerGridDefault) {
    // default constructed 2D grid
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> grid {};
    using nd_size_type = decltype(grid)::nd_size_type;
    using nd_value_type = decltype(grid)::nd_value_type;

    // basic grid info
    ASSERT_EQ(grid.dim(), 2);
    ASSERT_EQ(grid.size(), 4);
    check_range_equal(grid.shape(), nd_size_type { 2, 2 });
    check_range_near(grid.first(), nd_value_type { 0, 0 });
    check_range_near(grid.last(), nd_value_type { 1, 1 });

    // bin volum and center
    ASSERT_DOUBLE_EQ(grid.bin_volume(0, 0), 1.0);
    check_range_near(grid.center(0, 0), nd_value_type { 0.5, 0.5 });

    // grid points
    check_range_near(grid.at(0, 0), nd_value_type { 0, 0 });
    check_range_near(grid.at(0, 1), nd_value_type { 0, 1 });
    check_range_near(grid.at(1, 0), nd_value_type { 1, 0 });
    check_range_near(grid.at(1, 1), nd_value_type { 1, 1 });

    // bin index
    check_range_near(grid.index(0, 0), nd_value_type { 0, 0 });
    check_range_near(grid.index(0, 1), nd_value_type { 0, 1 });
    check_range_near(grid.index(1, 0), nd_value_type { 1, 0 });
    check_range_near(grid.index(1, 1), nd_value_type { 1, 1 });
    check_range_near(grid.index(0.1, 0.7), nd_value_type { 0, 0 });
}

// Test 2-dimensional linear grid.
TEST(SimplemcGrids, TwoDimensionalLinearGrid) {
    // grid #1 and #2
    simplemc::linear_grid g1 { 0, 20, 11 };

    // 2D grid
    simplemc::nd_grid grid { g1, g1 };
    using nd_size_type = decltype(grid)::nd_size_type;
    using nd_value_type = decltype(grid)::nd_value_type;
    ASSERT_EQ(grid.dim(), 2);
    ASSERT_EQ(grid.size(), g1.size() * g1.size());
    check_range_equal(grid.shape(), nd_size_type { g1.size(), g1.size() });
    check_range_near(grid.first(), nd_value_type { g1.first(), g1.first() });
    check_range_near(grid.last(), nd_value_type { g1.last(), g1.last() });

    // loop over 2D grid and its views
    using ranges::views::cartesian_product;
    using ranges::views::iota;
    auto view = grid.view();
    auto view_center = grid.view_center();
    auto view_vols = grid.view_bin_volumes();
    auto it_v = ranges::begin(view);
    auto it_vc = ranges::begin(view_center);
    auto it_vv = ranges::begin(view_vols);
    for (auto [i, j] : cartesian_product(iota(0, g1.size()), iota(0, g1.size()))) {
        nd_size_type idx_arr { i, j };
        nd_value_type val_arr { g1.at(i) + 0.1 * g1.bin_volume(i), g1.at(j) + 0.5 * g1.bin_volume(j) };
        check_range_near(grid.at(idx_arr), nd_value_type { g1.at(i), g1.at(j) });
        check_range_near(*it_v++, nd_value_type { g1.at(i), g1.at(j) });
        if (i < g1.size() - 1 && j < g1.size() - 1) {
            check_range_near(grid.center(idx_arr), nd_value_type { g1.center(i), g1.center(j) });
            check_range_near(*it_vc++, nd_value_type { g1.center(i), g1.center(j) });
            check_range_equal(grid.index(val_arr), idx_arr);
            ASSERT_DOUBLE_EQ(grid.bin_volume(idx_arr), g1.bin_volume(i) * g1.bin_volume(j));
            ASSERT_DOUBLE_EQ(*it_vv++, g1.bin_volume(i) * g1.bin_volume(j));
        }
    }

    // index subrange
    check_range_equal(grid.index_subrange(5, 2.2, 8.2), nd_size_type { 0, 2 });
    check_range_equal(grid.index_subrange(2, 10.5, 11.5), nd_size_type { 5, 5 });
}

// Test 3-dimensional grid.
TEST(SimplemcGrids, ThreeDimensionalGrid) {
    // grid #1 and #3
    std::vector<double> view1 { 3, 5, 11, 21 };
    std::vector<double> view_center1 { 4, 8, 16 };
    std::vector<double> view_vols1 { 2, 6, 10 };
    simplemc::power_grid g1 { 3, 21, 4, 2 };

    // grid #2
    std::vector<double> view2 { -3, -5, -11, -21 };
    std::vector<double> view_center2 { -4, -8, -16 };
    std::vector<double> view_vols2 { 2, 6, 10 };
    simplemc::power_grid g2 { -3, -21, 4, 2 };

    // 3D grid
    simplemc::nd_grid grid { g1, g2, g1 };
    using nd_size_type = decltype(grid)::nd_size_type;
    using nd_value_type = decltype(grid)::nd_value_type;
    ASSERT_EQ(grid.dim(), 3);
    ASSERT_EQ(grid.size(), g1.size() * g2.size() * g1.size());
    check_range_equal(grid.shape(), nd_size_type { g1.size(), g2.size(), g1.size() });
    check_range_near(grid.first(), nd_value_type { g1.first(), g2.first(), g1.first() });
    check_range_near(grid.last(), nd_value_type { g1.last(), g2.last(), g1.last() });

    // loop over 3D grid and its views
    using ranges::views::cartesian_product;
    using ranges::views::iota;
    auto view = grid.view();
    auto view_center = grid.view_center();
    auto view_vols = grid.view_bin_volumes();
    auto it_v = ranges::begin(view);
    auto it_vc = ranges::begin(view_center);
    auto it_vv = ranges::begin(view_vols);
    for (auto [i, j, k] : cartesian_product(iota(0, g1.size()), iota(0, g2.size()), iota(0, g1.size()))) {
        nd_size_type idx_arr { i, j, k };
        nd_value_type val_arr { view1[i] + 0.1 * view_vols1[i], view2[j] - 0.3 * view_vols2[j],
            view1[k] + 0.7 * view_vols1[k] };
        check_range_near(grid.at(idx_arr), nd_value_type { view1[i], view2[j], view1[k] });
        check_range_near(*it_v++, nd_value_type { view1[i], view2[j], view1[k] });
        if (i < g1.size() - 1 && j < g2.size() - 1 && k < g1.size() - 1) {
            check_range_near(grid.center(idx_arr), nd_value_type { view_center1[i], view_center2[j], view_center1[k] });
            check_range_near(*it_vc++, nd_value_type { view_center1[i], view_center2[j], view_center1[k] });
            check_range_equal(grid.index(val_arr), idx_arr);
            ASSERT_DOUBLE_EQ(grid.bin_volume(idx_arr), view_vols1[i] * view_vols2[j] * view_vols1[k]);
            ASSERT_DOUBLE_EQ(*it_vv++, view_vols1[i] * view_vols2[j] * view_vols1[k]);
        }
    }

    // index subrange
    check_range_equal(grid.index_subrange(3, 3.2, -12, 20), nd_size_type { 0, 1, 1 });
    check_range_equal(grid.index_subrange(2, 20.5, -4, 8), nd_size_type { 2, 0, 1 });
}
