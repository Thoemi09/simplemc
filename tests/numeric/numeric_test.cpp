/**
 * @file numeric_test.cpp
 * @brief Unit tests for simplemc-numeric.
 */

#include "../test_utils.hpp"

#include <simplemc/numeric/utils.hpp>
#include <simplemc/numeric/eigen.hpp>

#include <numbers>

// Test utilty functions.
TEST(SimplemcNumeric, UtilityFunctions) {
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(1e-5, 1e-5), 0);
    ASSERT_DOUBLE_EQ(simplemc::rel_diff(1e-5, 1e-5), 0);
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(1e-5, 2e-5), 1e-5);
    ASSERT_DOUBLE_EQ(simplemc::rel_diff(1e-5, 2e-5), 1);
    ASSERT_FALSE(simplemc::isfinite(1.0 / 0.0));
    std::complex<double> c1 { 1.0, std::log(-1) };
    ASSERT_FALSE(simplemc::isfinite(c1));
    ASSERT_EQ(simplemc::index_of_subset(4, 9, 1), 4);
    ASSERT_EQ(simplemc::index_of_subset(4, 9, 2), 3);
    ASSERT_EQ(simplemc::index_of_subset(4, 9, 3), 3);
    ASSERT_EQ(simplemc::index_of_subset(4, 9, 6), 1);
    ASSERT_EQ(simplemc::index_of_subset(4, 9, 8), 0);
    ASSERT_EQ(simplemc::index_of_subset(4, 9, 9), 0);
    ASSERT_EQ(simplemc::index_of_subset(7, 9, 1), 7);
    ASSERT_EQ(simplemc::index_of_subset(7, 9, 2), 6);
    ASSERT_EQ(simplemc::index_of_subset(7, 9, 3), 6);
    ASSERT_EQ(simplemc::index_of_subset(7, 9, 4), 5);
    ASSERT_EQ(simplemc::index_of_subset(7, 9, 7), 2);
    ASSERT_EQ(simplemc::index_of_subset(1, 9, 1), 1);
    ASSERT_EQ(simplemc::index_of_subset(1, 9, 2), 0);
    ASSERT_EQ(simplemc::index_of_subset(1, 9, 3), 0);
    ASSERT_EQ(simplemc::index_of_subset(1, 9, 4), 0);
    ASSERT_EQ(simplemc::index_of_subset(1, 9, 9), 0);
}

TEST(SimplemcNumeric, MapToInterval) {
    using std::numbers::pi;
    using namespace simplemc;
    double tol = 1e-10;
    double lb = -pi;
    double ub = pi;
    ASSERT_NEAR(0.0, map_to_interval(0.0, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(ub, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(lb, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(3 * pi, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(-3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, map_to_interval(-0.7 - 4 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7 + pi, map_to_interval(-0.7 - 3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, map_to_interval(-0.7 + 12 * pi, lb, ub), tol);
    ASSERT_NEAR(0.0, map_to_interval_lb(0.0, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(ub, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(lb, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(3 * pi, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(-3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, map_to_interval_lb(-0.7 - 4 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7 + pi, map_to_interval_lb(-0.7 - 3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, map_to_interval_lb(-0.7 + 12 * pi, lb, ub), tol);
    lb = 1.1;
    ub = 2.3;
    ASSERT_NEAR(1.7, map_to_interval(1.7, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(ub, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(lb, lb, ub), tol);
    ASSERT_NEAR(1.2, map_to_interval(1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(1.9, map_to_interval(1.9 - 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(1.7, map_to_interval_lb(1.7, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(ub, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(lb, lb, ub), tol);
    ASSERT_NEAR(1.2, map_to_interval_lb(1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(1.9, map_to_interval_lb(1.9 - 3 * 1.2, lb, ub), tol);
    lb = -2.3;
    ub = -1.1;
    ASSERT_NEAR(-1.7, map_to_interval(-1.7, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(ub, lb, ub), tol);
    ASSERT_NEAR(ub, map_to_interval(lb, lb, ub), tol);
    ASSERT_NEAR(-1.2, map_to_interval(-1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(-1.9, map_to_interval(-1.9 - 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(-1.7, map_to_interval_lb(-1.7, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(ub, lb, ub), tol);
    ASSERT_NEAR(lb, map_to_interval_lb(lb, lb, ub), tol);
    ASSERT_NEAR(-1.2, map_to_interval_lb(-1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(-1.9, map_to_interval_lb(-1.9 - 3 * 1.2, lb, ub), tol);
}

TEST(SimplemcNumeric, WithinBounds) {
    double min_diff = 1e-3;
    double low = 4.2;
    double up = 5.0;
    double val_in1 = 4.5;
    double val_in2 = low + 1e-4;
    double val_in3 = up - 1e-4;
    ASSERT_TRUE(simplemc::within_bounds(val_in1, low, up, min_diff));
    ASSERT_FALSE(simplemc::within_bounds(val_in2, low, up, min_diff));
    ASSERT_FALSE(simplemc::within_bounds(val_in3, low, up, min_diff));
}

TEST(SimplemcNumerics, PolarCartesianConversion) {
    simplemc::vector<3>::type vec { 1.0, 0.0, 0.0 };
    simplemc::vector<3>::type exp { 1.0, std::numbers::pi / 2, 0.0 };
    check_range_near(simplemc::cartesian_to_polar(vec), exp);
    simplemc::vector<2>::type vec2 { std::numbers::sqrt2, std::numbers::pi / 4 };
    simplemc::vector<2>::type exp2 { 1.0, 1.0 };
    check_range_near(simplemc::polar_to_cartesian(vec2), exp2);
}
