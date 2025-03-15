#include "../test_utils.hpp"

#include <fmt/base.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric/polynomial_interpolation.hpp>
#include <simplemc/numeric/quadrature.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

// Test basic quadrature class.
TEST(SimplemcNumeric, BasicQuadrature) {
    using std::numbers::pi;
    simplemc::basic_quadrature bq([](double x) { return std::sin(x); }, 0, pi);
    ASSERT_EQ(bq.integration_limits(), (std::array { 0.0, pi }));
    ASSERT_EQ(bq.level(), 1);
    ASSERT_DOUBLE_EQ(bq.current(), (std::sin(pi) - std::sin(0)) * 0.5 * pi);
    double current = bq.current();
    double last = 0.0;
    for (int i = 0; i < 20; ++i) {
        bq.next();
        ASSERT_EQ(bq.last(), current);
        last = current;
        current = bq.current();
        ASSERT_EQ(std::abs(current - last) <= 1e-5 * std::abs(current), bq.is_converged(1e-5));
        ASSERT_EQ(bq.level(), i + 2);
    }
    ASSERT_NEAR(bq.current(), 2.0, 1e-10);
}

// Test trapezoidal quadrature.
TEST(SimplemcNumeric, TrapzoidalQuadrature) {
    auto [res, err, n] = simplemc::trapez_quadrature([](double x) { return std::sin(x); }, 0, std::numbers::pi, 1e-10);
    ASSERT_NEAR(res, 2.0, 2.0 * 1e-10);
    ASSERT_TRUE(err < 1e-10);
    fmt::print("Number of iterations: {}\n", n);
}

// Test Simpson's quadrature.
TEST(SimplemcNumeric, SimpsonQuadrature) {
    auto [res, err, n] = simplemc::simpson_quadrature([](double x) { return std::sin(x); }, 0, std::numbers::pi, 1e-10);
    ASSERT_NEAR(res, 2.0, 2.0 * 1e-10);
    ASSERT_TRUE(err < 1e-10);
    fmt::print("Number of iterations: {}\n", n);
}

// Test a combination of interpolation and quadrature.
TEST(SimplemcNumeric, InterpolationWithQuadrature) {
    long nx = 50;
    simplemc::linear_grid grid(0.0, std::numbers::pi, nx);
    std::vector<double> fvals(nx);
    for (long i = 0; i < nx; ++i) {
        fvals[i] = std::sin(grid.at(i));
    }
    simplemc::polynomial_interpolation pi(grid, fvals, 2);
    auto func = [&pi](double x) { return pi(x); };
    auto [res, err, n] = simplemc::simpson_quadrature(func, grid.first(), grid.last());
    ASSERT_NEAR(res, 2.0, 1e-6);
    ASSERT_TRUE(err < 1e-7);
    fmt::print("Number of iterations: {}\n", n);
}
