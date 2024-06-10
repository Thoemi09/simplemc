/**
 * @file
 * @brief Unit tests for simplemc-grids.
 */

#include "../test_utils.hpp"

#include <simplemc/grids.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <vector>

// Test grid size.
TEST(SimplemcGrids, GridSizeCheck) {
    ASSERT_THROW(simplemc::linear_grid(0.0, 1.0, 0), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_grid(0.0, 1.0, -1), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_grid(0.0, 1.0, 1), simplemc::simplemc_exception);
}

// Test increasing linear grid.
TEST(SimplemcGrids, LinearGridAscending) {
    simplemc::linear_grid lg { 0, 10, 11 };
    ASSERT_EQ(lg.size(), 11);
    ASSERT_EQ(lg.first(), 0);
    ASSERT_EQ(lg.last(), 10);
    for (int i = 0; i < lg.size() - 1; ++i) {
        ASSERT_DOUBLE_EQ(lg.at(i), i);
        ASSERT_DOUBLE_EQ(lg.center(i), i + 0.5);
        ASSERT_DOUBLE_EQ(lg.bin_volume(i), 1);
    }
    ASSERT_DOUBLE_EQ(lg.at(10), 10);
    ASSERT_EQ(lg.index(0.0), 0);
    ASSERT_EQ(lg.index(9.9), 9);
    ASSERT_EQ(lg.index(2.5), 2);
    check_range_near(lg.view(), ranges::iota_view(0, 11));
    check_range_near(
        lg.view_center(), ranges::iota_view(0, 10) | ranges::views::transform([](auto i) { return i + 0.5; }));
    check_range_near(lg.view_bin_volumes(), ranges::views::repeat(1.0) | ranges::views::take(10));
}

// Test decreasing linear grid.
TEST(SimplemcGrids, LinearGridDescending) {
    simplemc::linear_grid lg { 0, -10, 11 };
    ASSERT_EQ(lg.size(), 11);
    ASSERT_EQ(lg.first(), 0);
    ASSERT_EQ(lg.last(), -10);
    for (int i = 0; i < lg.size() - 1; ++i) {
        ASSERT_DOUBLE_EQ(lg.at(i), -i);
        ASSERT_DOUBLE_EQ(lg.center(i), -i - 0.5);
        ASSERT_DOUBLE_EQ(lg.bin_volume(i), 1);
    }
    ASSERT_DOUBLE_EQ(lg.at(10), -10);
    ASSERT_EQ(lg.index(0.0), 0);
    ASSERT_EQ(lg.index(-9.9), 9);
    ASSERT_EQ(lg.index(-2.5), 2);
    check_range_near(lg.view(), ranges::iota_view(0, 11) | ranges::views::transform([](auto i) { return -i; }));
    check_range_near(
        lg.view_center(), ranges::iota_view(0, 10) | ranges::views::transform([](auto i) { return -i - 0.5; }));
    check_range_near(lg.view_bin_volumes(), ranges::views::repeat(1.0) | ranges::views::take(10));
}

// Test increasing power grid.
TEST(SimplemcGrids, PowerGridAscending) {
    ASSERT_THROW(simplemc::power_grid(3, 5, 11, -1.0), simplemc::simplemc_exception);
    std::vector<double> exp_at { 3, 5, 11, 21 };
    std::vector<double> exp_center { 4, 8, 16 };
    std::vector<double> exp_vol { 2, 6, 10 };
    simplemc::power_grid pg { 3, 21, 4, 2 };
    ASSERT_EQ(pg.size(), 4);
    ASSERT_EQ(pg.first(), 3);
    ASSERT_EQ(pg.last(), 21);
    for (int i = 0; i < pg.size() - 1; ++i) {
        ASSERT_DOUBLE_EQ(pg.at(i), exp_at[i]);
        ASSERT_DOUBLE_EQ(pg.center(i), exp_center[i]);
        ASSERT_DOUBLE_EQ(pg.bin_volume(i), exp_vol[i]);
    }
    ASSERT_DOUBLE_EQ(pg.at(3), exp_at.back());
    ASSERT_EQ(pg.index(3.0), 0);
    ASSERT_EQ(pg.index(20), 2);
    ASSERT_EQ(pg.index(7.0), 1);
    check_range_near(pg.view(), exp_at);
    check_range_near(pg.view_center(), exp_center);
    check_range_near(pg.view_bin_volumes(), exp_vol);
}

// Test decreasing power grid.
TEST(SimplemcGrids, PowerGridDescending) {
    std::vector<double> exp_at { -3, -5, -11, -21 };
    std::vector<double> exp_center { -4, -8, -16 };
    std::vector<double> exp_vol { 2, 6, 10 };
    simplemc::power_grid pg { -3, -21, 4, 2 };
    ASSERT_EQ(pg.size(), 4);
    ASSERT_EQ(pg.first(), -3);
    ASSERT_EQ(pg.last(), -21);
    for (int i = 0; i < pg.size() - 1; ++i) {
        ASSERT_DOUBLE_EQ(pg.at(i), exp_at[i]);
        ASSERT_DOUBLE_EQ(pg.center(i), exp_center[i]);
        ASSERT_DOUBLE_EQ(pg.bin_volume(i), exp_vol[i]);
    }
    ASSERT_DOUBLE_EQ(pg.at(3), exp_at.back());
    ASSERT_EQ(pg.index(-3.0), 0);
    ASSERT_EQ(pg.index(-20), 2);
    ASSERT_EQ(pg.index(-7.0), 1);
    check_range_near(pg.view(), exp_at);
    check_range_near(pg.view_center(), exp_center);
    check_range_near(pg.view_bin_volumes(), exp_vol);
}

// Test increasing symmetric power grid.
TEST(SimplemcGrids, SymmetricPowerGridIncreasing) {
    ASSERT_THROW(simplemc::symmetric_power_grid(3, 5, 10, 1.0), simplemc::simplemc_exception);
    double begin = -10;
    double end = 10;
    double mid = 0.0;
    double power = 2.0;
    long size = 21;
    long mid_idx = static_cast<long>(size / 2) + 1;
    simplemc::power_grid g1 { begin, mid, mid_idx, power };
    simplemc::power_grid g2 { end, mid, mid_idx, power };
    simplemc::symmetric_power_grid grid { begin, end, size, power };
    check_range_near(grid.grid1().view(), g1.view());
    check_range_near(grid.grid2().view(), g2.view());
    ASSERT_EQ(grid.first(), begin);
    ASSERT_EQ(grid.last(), end);
    ASSERT_EQ(grid.midpoint(), mid);
    ASSERT_EQ(grid.grid1().power(), g1.power());
    ASSERT_EQ(grid.grid1().scale(), g1.scale());
    ASSERT_EQ(grid.grid2().power(), g2.power());
    ASSERT_EQ(grid.grid2().scale(), g2.scale());
    ASSERT_EQ(grid.size(), size);
    ASSERT_DOUBLE_EQ(grid.at(5), g1.at(5));
    ASSERT_DOUBLE_EQ(grid.at(19), g2.at(1));
    ASSERT_DOUBLE_EQ(grid.center(8), g1.center(8));
    ASSERT_DOUBLE_EQ(grid.center(10), g2.center(9));
    ASSERT_DOUBLE_EQ(grid.bin_volume(9), g1.bin_volume(9));
    ASSERT_DOUBLE_EQ(grid.bin_volume(19), g2.bin_volume(0));
    ASSERT_DOUBLE_EQ(grid.bin_volume(2), grid.bin_volume(17));
    simplemc::symmetric_power_grid grid2 { 3, 5, 11, 1.0 };
    grid2.reset(begin, end, size, power);
    check_range_near(grid2.grid1().view(), g1.view());
    check_range_near(grid2.grid2().view(), g2.view());
    check_range_near(
        grid2.view(), ranges::concat_view(g1.view(), ranges::drop_view(ranges::reverse_view(g2.view()), 1)));
    check_range_near(
        grid2.view_center(), ranges::concat_view(g1.view_center(), ranges::reverse_view(g2.view_center())));
    check_range_near(grid2.view_bin_volumes(),
        ranges::concat_view(g1.view_bin_volumes(), ranges::reverse_view(g2.view_bin_volumes())));
}

// Test decreasing symmetric power grid.
TEST(SimplemcGrids, SymmetricPowerGridDecreasing) {
    ASSERT_THROW(simplemc::symmetric_power_grid(5, 3, 10, 1.0), simplemc::simplemc_exception);
    double begin = 10;
    double end = -10;
    double mid = 0.0;
    double power = 2.0;
    long size = 21;
    long mid_idx = static_cast<long>(size / 2) + 1;
    simplemc::power_grid g1 { begin, mid, mid_idx, power };
    simplemc::power_grid g2 { end, mid, mid_idx, power };
    simplemc::symmetric_power_grid grid { begin, end, size, power };
    check_range_near(grid.grid1().view(), g1.view());
    check_range_near(grid.grid2().view(), g2.view());
    ASSERT_EQ(grid.first(), begin);
    ASSERT_EQ(grid.last(), end);
    ASSERT_EQ(grid.midpoint(), mid);
    ASSERT_EQ(grid.grid1().power(), g1.power());
    ASSERT_EQ(grid.grid1().scale(), g1.scale());
    ASSERT_EQ(grid.grid2().power(), g2.power());
    ASSERT_EQ(grid.grid2().scale(), g2.scale());
    ASSERT_EQ(grid.size(), size);
    ASSERT_DOUBLE_EQ(grid.at(5), g1.at(5));
    ASSERT_DOUBLE_EQ(grid.at(19), g2.at(1));
    ASSERT_DOUBLE_EQ(grid.center(8), g1.center(8));
    ASSERT_DOUBLE_EQ(grid.center(10), g2.center(9));
    ASSERT_DOUBLE_EQ(grid.bin_volume(9), g1.bin_volume(9));
    ASSERT_DOUBLE_EQ(grid.bin_volume(19), g2.bin_volume(0));
    ASSERT_DOUBLE_EQ(grid.bin_volume(2), grid.bin_volume(17));
    simplemc::symmetric_power_grid grid2 { 3, 5, 11, 1.0 };
    grid2.reset(begin, end, size, power);
    check_range_near(grid2.grid1().view(), g1.view());
    check_range_near(grid2.grid2().view(), g2.view());
    check_range_near(
        grid2.view(), ranges::concat_view(g1.view(), ranges::drop_view(ranges::reverse_view(g2.view()), 1)));
    check_range_near(
        grid2.view_center(), ranges::concat_view(g1.view_center(), ranges::reverse_view(g2.view_center())));
    check_range_near(grid2.view_bin_volumes(),
        ranges::concat_view(g1.view_bin_volumes(), ranges::reverse_view(g2.view_bin_volumes())));
}

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
