// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../test_utils.hpp"

#include <simplemc/grids/concepts.hpp>
#include <simplemc/grids/custom_grid.hpp>
#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/nd_grid.hpp>
#include <simplemc/grids/power_grid.hpp>
#include <simplemc/grids/symmetric_power_grid.hpp>
#include <simplemc/grids/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <vector>

namespace rgs = simplemc::ranges;

// Test concept.
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::linear_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::power_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::linear_grid, simplemc::linear_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::power_grid, simplemc::power_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::symmetric_power_grid, simplemc::linear_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::custom_grid, simplemc::power_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid, simplemc::custom_grid>>);
static_assert(simplemc::grid_nd<simplemc::nd_grid<simplemc::symmetric_power_grid, simplemc::linear_grid,
        simplemc::power_grid, simplemc::custom_grid>>);

// Test 2-dimensional default constructed linear - power grid.
TEST(SimplemcGrids, TwoDimensionalLinearPowerGridDefault) {
    // default constructed 2D grid
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> grid {};
    using size_type = decltype(grid)::size_type;
    using value_type = decltype(grid)::value_type;

    // basic grid info
    ASSERT_EQ(grid.dim(), 2);
    ASSERT_EQ(grid.size(), 4);
    check_equal(grid.shape(), size_type { 2, 2 });
    check_near(grid.first(), value_type { 0, 0 });
    check_near(grid.last(), value_type { 1, 1 });

    // bin volum and center
    ASSERT_DOUBLE_EQ(grid.volume(0, 0), 1.0);
    check_near(grid.center(0, 0), value_type { 0.5, 0.5 });

    // grid points
    check_near(grid.at(0, 0), value_type { 0, 0 });
    check_near(grid.at(0, 1), value_type { 0, 1 });
    check_near(grid.at(1, 0), value_type { 1, 0 });
    check_near(grid.at(1, 1), value_type { 1, 1 });

    // bin index
    check_near(grid.index(0, 0), value_type { 0, 0 });
    check_near(grid.index(0, 1), value_type { 0, 1 });
    check_near(grid.index(1, 0), value_type { 1, 0 });
    check_near(grid.index(1, 1), value_type { 1, 1 });
    check_near(grid.index(0.1, 0.7), value_type { 0, 0 });
}

// Test constexpr functionality of nd_grid.
TEST(SimplemcGrids, TwoDimensionalLinearGridConstexpr) {
    using size_type = simplemc::nd_grid<simplemc::linear_grid, simplemc::linear_grid>::size_type;
    using value_type = simplemc::nd_grid<simplemc::linear_grid, simplemc::linear_grid>::value_type;

    constexpr simplemc::linear_grid g1 { 0.0, 10.0, 11 };
    constexpr simplemc::linear_grid g2 { 0.0, 5.0, 6 };
    constexpr simplemc::nd_grid grid { g1, g2 };

    static_assert(grid.dim() == 2);
    static_assert(grid.size() == 11L * 6L);
    static_assert(grid.first() == value_type { 0.0, 0.0 });
    static_assert(grid.last() == value_type { 10.0, 5.0 });
    static_assert(grid.shape() == size_type { 11, 6 });
    static_assert(grid.at(0, 0) == value_type { 0.0, 0.0 });
    static_assert(grid.at(5, 3) == value_type { 5.0, 3.0 });
    static_assert(grid.at(10, 5) == value_type { 10.0, 5.0 });
    static_assert(grid.center(0, 0) == value_type { 0.5, 0.5 });
    static_assert(grid.center(5, 3) == value_type { 5.5, 3.5 });
    static_assert(grid.index(0.5, 0.5) == size_type { 0, 0 });
    static_assert(grid.index(5.5, 3.5) == size_type { 5, 3 });
    static_assert(grid.index(9.9, 4.9) == size_type { 9, 4 });

    // runtime checks for volume (std::abs may not be constexpr-friendly on all compilers)
    ASSERT_DOUBLE_EQ(grid.volume(0, 0), 1.0);
    ASSERT_DOUBLE_EQ(grid.volume(5, 3), 1.0);
}

// Test 2-dimensional linear grid.
TEST(SimplemcGrids, TwoDimensionalLinearGrid) {
    // grid #1 and #2
    simplemc::linear_grid g1 { 0, 20, 11 };

    // 2D grid
    simplemc::nd_grid grid { g1, g1 };
    using size_type = decltype(grid)::size_type;
    using value_type = decltype(grid)::value_type;
    ASSERT_EQ(grid.dim(), 2);
    ASSERT_EQ(grid.size(), g1.size() * g1.size());
    check_equal(grid.shape(), size_type { g1.size(), g1.size() });
    check_near(grid.first(), value_type { g1.first(), g1.first() });
    check_near(grid.last(), value_type { g1.last(), g1.last() });

    // loop over 2D grid and its views
    using rgs::views::cartesian_product;
    using rgs::views::iota;
    auto view = simplemc::grid_view(grid);
    auto view_center = simplemc::bin_center_view(grid);
    auto view_vols = simplemc::bin_volume_view(grid);
    auto it_v = rgs::begin(view);
    auto it_vc = rgs::begin(view_center);
    auto it_vv = rgs::begin(view_vols);
    for (auto [i, j] : cartesian_product(iota(0, g1.size()), iota(0, g1.size()))) {
        size_type idx_arr { i, j };
        check_near(grid.at(idx_arr), value_type { g1.at(i), g1.at(j) });
        check_near(*it_v++, value_type { g1.at(i), g1.at(j) });
        if (i < g1.size() - 1 && j < g1.size() - 1) {
            value_type val_arr { g1.at(i) + 0.1 * g1.volume(i), g1.at(j) + 0.5 * g1.volume(j) };
            check_near(grid.center(idx_arr), value_type { g1.center(i), g1.center(j) });
            check_near(*it_vc++, value_type { g1.center(i), g1.center(j) });
            check_equal(grid.index(val_arr), idx_arr);
            ASSERT_DOUBLE_EQ(grid.volume(idx_arr), g1.volume(i) * g1.volume(j));
            ASSERT_DOUBLE_EQ(*it_vv++, g1.volume(i) * g1.volume(j));
        }
    }

    // index subrange
    check_equal(simplemc::index_subrange(grid, 5, 2.2, 8.2), size_type { 0, 2 });
    check_equal(simplemc::index_subrange(grid, 2, 10.5, 11.5), size_type { 5, 5 });

    // grid as a range vs explicit view
    check_near(grid, view);
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
    using size_type = decltype(grid)::size_type;
    using value_type = decltype(grid)::value_type;
    ASSERT_EQ(grid.dim(), 3);
    ASSERT_EQ(grid.size(), g1.size() * g2.size() * g1.size());
    check_equal(grid.shape(), size_type { g1.size(), g2.size(), g1.size() });
    check_near(grid.first(), value_type { g1.first(), g2.first(), g1.first() });
    check_near(grid.last(), value_type { g1.last(), g2.last(), g1.last() });

    // loop over 3D grid and its views
    using rgs::views::cartesian_product;
    using rgs::views::iota;
    auto view = simplemc::grid_view(grid);
    auto view_center = simplemc::bin_center_view(grid);
    auto view_vols = simplemc::bin_volume_view(grid);
    auto it_v = rgs::begin(view);
    auto it_vc = rgs::begin(view_center);
    auto it_vv = rgs::begin(view_vols);
    for (auto [i, j, k] : cartesian_product(iota(0, g1.size()), iota(0, g2.size()), iota(0, g1.size()))) {
        size_type idx_arr { i, j, k };
        check_near(grid.at(idx_arr), value_type { view1[i], view2[j], view1[k] });
        check_near(*it_v++, value_type { view1[i], view2[j], view1[k] });
        if (i < g1.size() - 1 && j < g2.size() - 1 && k < g1.size() - 1) {
            value_type val_arr { view1[i] + 0.1 * view_vols1[i], view2[j] - 0.3 * view_vols2[j],
                view1[k] + 0.7 * view_vols1[k] };
            check_near(grid.center(idx_arr), value_type { view_center1[i], view_center2[j], view_center1[k] });
            check_near(*it_vc++, value_type { view_center1[i], view_center2[j], view_center1[k] });
            check_equal(grid.index(val_arr), idx_arr);
            ASSERT_DOUBLE_EQ(grid.volume(idx_arr), view_vols1[i] * view_vols2[j] * view_vols1[k]);
            ASSERT_DOUBLE_EQ(*it_vv++, view_vols1[i] * view_vols2[j] * view_vols1[k]);
        }
    }

    // index subrange
    check_equal(simplemc::index_subrange(grid, 3, 3.2, -12, 20), size_type { 0, 1, 1 });
    check_equal(simplemc::index_subrange(grid, 2, 20.5, -4, 8), size_type { 2, 0, 1 });

    // grid as a range vs explicit view
    check_near(grid, view);
}
