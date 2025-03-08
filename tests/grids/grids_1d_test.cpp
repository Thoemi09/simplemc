#include "../test_utils.hpp"

#include <simplemc/grids/concepts.hpp>
#include <simplemc/grids/custom_grid.hpp>
#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/power_grid.hpp>
#include <simplemc/grids/symmetric_power_grid.hpp>
#include <simplemc/grids/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cstddef>
#include <vector>

namespace rgs = simplemc::ranges;

// Concatenate 2 views.
template <typename R1, typename R2>
auto concat(R1&& view1, R2&& view2) { // NOLINT (ranges need not be forwarded)
    std::vector<rgs::range_value_t<R1>> vec {};
    vec.reserve(static_cast<std::size_t>(rgs::size(view1) + rgs::size(view2)));
    for (auto x : view1) {
        vec.push_back(x);
    }
    for (auto x : view2) {
        vec.push_back(x);
    }
    return vec;
}

// Test 1d grids.
void test_1d_grids(const auto& g, auto a, auto b, auto&& view, auto&& view_center, auto&& view_vols) {
    // basic grid info
    ASSERT_EQ(g.size(), rgs::size(view));
    ASSERT_EQ(g.first(), a);
    ASSERT_EQ(g.last(), b);

    // loop over bins and grid points
    double fac = (a < b ? 1 : -1);
    for (int i = 0; auto [pt, c, vol] : rgs::views::zip(view, view_center, view_vols)) {
        ASSERT_DOUBLE_EQ(g.at(i), pt);
        ASSERT_DOUBLE_EQ(g.center(i), c);
        ASSERT_DOUBLE_EQ(g.volume(i), vol);
        ASSERT_EQ(g.index(pt + fac * 1e-10), i);
        ASSERT_EQ(g.index(pt + fac * 0.3 * vol), i);
        ++i;
    }
    ASSERT_DOUBLE_EQ(g.at(g.size() - 1), b);

    // views
    check_range_near(simplemc::grid_view(g), view);
    check_range_near(simplemc::bin_center_view(g), view_center);
    check_range_near(simplemc::bin_volume_view(g), view_vols);
}

// Test concept.
static_assert(simplemc::grid_1d<simplemc::linear_grid>);
static_assert(simplemc::grid_1d<simplemc::power_grid>);
static_assert(simplemc::grid_1d<simplemc::symmetric_power_grid>);
static_assert(simplemc::grid_1d<simplemc::custom_grid>);

// Test incorrect grid sizes.
TEST(SimplemcGrids, GridSizeCheck) {
    ASSERT_THROW(simplemc::linear_grid(0.0, 1.0, 0), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_grid(0.0, 1.0, -1), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_grid(0.0, 1.0, 1), simplemc::simplemc_exception);
}

// Test index_subrange on increasing grids.
TEST(SimplemcGrids, IndexSubrangeIncreasingGrid) {
    using namespace simplemc;
    linear_grid lg { 0, 10, 11 };
    ASSERT_EQ(simplemc::index_subrange(lg, 2, 0.5), 0);
    ASSERT_EQ(simplemc::index_subrange(lg, 3, 1.5), 0);
    ASSERT_EQ(simplemc::index_subrange(lg, 3, 2.5), 1);
    ASSERT_EQ(simplemc::index_subrange(lg, 4, 2.5), 1);
    ASSERT_EQ(simplemc::index_subrange(lg, 5, 2.5), 0);
    ASSERT_EQ(simplemc::index_subrange(lg, 4, 9.5), 7);
    ASSERT_EQ(simplemc::index_subrange(lg, 3, 9.5), 8);
    ASSERT_EQ(simplemc::index_subrange(lg, 2, 9.5), 9);
}

// Test index_subrange on decreasing grids.
TEST(SimplemcGrids, IndexSubrangeDecreasingGrid) {
    simplemc::linear_grid lg { 10, 0, 11 };
    ASSERT_EQ(simplemc::index_subrange(lg, 2, 0.5), 9);
    ASSERT_EQ(simplemc::index_subrange(lg, 3, 1.5), 7);
    ASSERT_EQ(simplemc::index_subrange(lg, 3, 2.5), 6);
    ASSERT_EQ(simplemc::index_subrange(lg, 4, 2.5), 6);
    ASSERT_EQ(simplemc::index_subrange(lg, 5, 2.5), 5);
    ASSERT_EQ(simplemc::index_subrange(lg, 4, 8.5), 0);
    ASSERT_EQ(simplemc::index_subrange(lg, 3, 8.5), 0);
    ASSERT_EQ(simplemc::index_subrange(lg, 2, 8.5), 1);
}

// Test default linear grid.
TEST(SimplemcGrids, LinearGridDefault) {
    simplemc::linear_grid lg {};
    auto view = std::vector<double> { 0.0, 1.0 };
    auto view_center = std::vector<double> { 0.5 };
    auto view_vols = std::vector<double> { 1.0 };
    test_1d_grids(lg, 0, 1, view, view_center, view_vols);
}

// Test increasing linear grid.
TEST(SimplemcGrids, LinearGridIncreasing) {
    simplemc::linear_grid lg { 0, 10, 11 };
    auto view = rgs::iota_view(0, 11);
    auto view_center = rgs::iota_view(0, 10) | rgs::views::transform([](auto i) { return i + 0.5; });
    auto view_vols = rgs::views::repeat(1.0) | rgs::views::take(10);
    test_1d_grids(lg, 0, 10, view, view_center, view_vols);
}

// Test decreasing linear grid.
TEST(SimplemcGrids, LinearGridDescending) {
    simplemc::linear_grid lg { 0, -10, 11 };
    auto view = rgs::iota_view(0, 11) | rgs::views::transform([](auto i) { return -i; });
    auto view_center = rgs::iota_view(0, 10) | rgs::views::transform([](auto i) { return -i - 0.5; });
    auto view_vols = rgs::views::repeat(1.0) | rgs::views::take(10);
    test_1d_grids(lg, 0, -10, view, view_center, view_vols);
}

// Test default power grid.
TEST(SimplemcGrids, PowerGridDefault) {
    simplemc::linear_grid pg {};
    auto view = std::vector<double> { 0.0, 1.0 };
    auto view_center = std::vector<double> { 0.5 };
    auto view_vols = std::vector<double> { 1.0 };
    test_1d_grids(pg, 0, 1, view, view_center, view_vols);
}

// Test increasing power grid.
TEST(SimplemcGrids, PowerGridIncreasing) {
    // wrong power parameter
    ASSERT_THROW(simplemc::power_grid(3, 5, 11, -1.0), simplemc::simplemc_exception);

    // test grid
    std::vector<double> view { 3, 5, 11, 21 };
    std::vector<double> view_center { 4, 8, 16 };
    std::vector<double> view_vols { 2, 6, 10 };
    simplemc::power_grid pg { 3, 21, 4, 2 };
    test_1d_grids(pg, 3, 21, view, view_center, view_vols);
}

// Test decreasing power grid.
TEST(SimplemcGrids, PowerGridDescending) {
    std::vector<double> view { -3, -5, -11, -21 };
    std::vector<double> view_center { -4, -8, -16 };
    std::vector<double> view_vols { 2, 6, 10 };
    simplemc::power_grid pg { -3, -21, 4, 2 };
    test_1d_grids(pg, -3, -21, view, view_center, view_vols);
}

// Test default symmetric power grid.
TEST(SimplemcGrids, SymmetricPowerGridDefault) {
    simplemc::symmetric_power_grid pg {};
    auto view = std::vector<double> { 0.0, 0.5, 1.0 };
    auto view_center = std::vector<double> { 0.25, 0.75 };
    auto view_vols = std::vector<double> { 0.5, 0.5 };
    test_1d_grids(pg, 0, 1, view, view_center, view_vols);
}

// Test increasing symmetric power grid.
TEST(SimplemcGrids, SymmetricPowerGridIncreasing) {
    using namespace simplemc;

    // wrong grid size
    ASSERT_THROW(symmetric_power_grid(3, 5, 10, 1.0), simplemc_exception);

    // test grid
    double begin = -10;
    double end = 10;
    double mid = 0.0;
    double power = 2.0;
    long size = 21;
    long mid_idx = static_cast<long>(size / 2) + 1;
    power_grid g1 { begin, mid, mid_idx, power };
    power_grid g2 { end, mid, mid_idx, power };
    symmetric_power_grid pg { begin, end, size, power };
    check_range_near(grid_view(pg.grid1()), grid_view(g1));
    check_range_near(grid_view(pg.grid2()), grid_view(g2));
    auto view = concat(grid_view(g1), rgs::drop_view(rgs::reverse_view(grid_view(g2)), 1));
    auto view_center = concat(bin_center_view(g1), rgs::reverse_view(bin_center_view(g2)));
    auto view_vols = concat(bin_volume_view(g1), rgs::reverse_view(bin_volume_view(g2)));
    test_1d_grids(pg, begin, end, view, view_center, view_vols);
}

// Test decreasing symmetric power grid.
TEST(SimplemcGrids, SymmetricPowerGridDecreasing) {
    // wrong grid size
    ASSERT_THROW(simplemc::symmetric_power_grid(5, 3, 10, 1.0), simplemc::simplemc_exception);

    // test grid
    double begin = 10;
    double end = -10;
    double mid = 0.0;
    double power = 2.0;
    long size = 21;
    long mid_idx = static_cast<long>(size / 2) + 1;
    simplemc::power_grid g1 { begin, mid, mid_idx, power };
    simplemc::power_grid g2 { end, mid, mid_idx, power };
    simplemc::symmetric_power_grid pg { begin, end, size, power };
    check_range_near(grid_view(pg.grid1()), grid_view(g1));
    check_range_near(grid_view(pg.grid2()), grid_view(g2));
    auto view = concat(grid_view(g1), rgs::drop_view(rgs::reverse_view(grid_view(g2)), 1));
    auto view_center = concat(bin_center_view(g1), rgs::reverse_view(bin_center_view(g2)));
    auto view_vols = concat(bin_volume_view(g1), rgs::reverse_view(bin_volume_view(g2)));
    test_1d_grids(pg, begin, end, view, view_center, view_vols);
}

// Test default custom grid.
TEST(SimplemcGrids, CustomGridDefault) {
    simplemc::custom_grid cg {};
    auto view = std::vector<double> { 0.0, 1.0 };
    auto view_center = std::vector<double> { 0.5 };
    auto view_vols = std::vector<double> { 1.0 };
    test_1d_grids(cg, 0, 1, view, view_center, view_vols);
}

// Test increasing custom grid.
TEST(SimplemcGrids, CustomGridIncreasing) {
    auto view = std::vector<double> { 1.0, 1.1, 2.4, 5.6, 12.3, 100.0 };
    auto view_center = rgs::iota_view(1ul, view.size()) |
        rgs::views::transform([&view](auto i) { return 0.5 * (view[i] + view[i - 1]); });
    auto view_vols =
        rgs::iota_view(1ul, view.size()) | rgs::views::transform([&view](auto i) { return view[i] - view[i - 1]; });
    simplemc::custom_grid cg { view };
    test_1d_grids(cg, view.front(), view.back(), view, view_center, view_vols);
}

// Test decreasing custom grid.
TEST(SimplemcGrids, CustomGridDecreasing) {
    auto view = std::vector<double> { 100.0, 12.3, 5.6, 2.4, 1.1, 1.0 };
    auto view_center = rgs::iota_view(1ul, view.size()) |
        rgs::views::transform([&view](auto i) { return 0.5 * (view[i] + view[i - 1]); });
    auto view_vols =
        rgs::iota_view(1ul, view.size()) | rgs::views::transform([&view](auto i) { return view[i - 1] - view[i]; });
    simplemc::custom_grid cg { view };
    test_1d_grids(cg, view.front(), view.back(), view, view_center, view_vols);
}
