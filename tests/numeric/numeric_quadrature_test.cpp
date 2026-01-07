#include "../test_utils.hpp"

#include <fmt/base.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric/polynomial_interpolation.hpp>
#include <simplemc/numeric/quadrature.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

// Test basic quadrature class.
TEST(SimplemcNumeric, BasicQuadrature) {
    using std::numbers::pi;
    simplemc::basic_quadrature bq { [](double x) { return std::sin(x); }, 0, pi };

    // accessors
    ASSERT_EQ(bq.integration_limits(), (std::array { 0.0, pi }));
    ASSERT_EQ(bq.level(), 1);
    ASSERT_DOUBLE_EQ(bq.integrand()(pi / 2), 1.0);
    ASSERT_DOUBLE_EQ(bq.current(), (std::sin(pi) - std::sin(0)) * 0.5 * pi);

    // iterate and check convergence
    double current = bq.current();
    double last = 0.0;
    for (int i = 0; i < 20; ++i) {
        auto tmp = bq.next();
        ASSERT_EQ(bq.last(), current);
        last = current;
        current = bq.current();
        ASSERT_EQ(tmp, current);
        ASSERT_EQ(std::abs(current - last) <= 1e-5 * std::abs(current), bq.is_converged(1e-5));
        ASSERT_EQ(bq.level(), i + 2);
    }
    ASSERT_NEAR(bq.current(), 2.0, 1e-10);
}

// Test basic_quadrature with std::function.
TEST(SimplemcNumeric, BasicQuadratureStdFunction) {
    std::function<double(double)> f = [](double x) { return std::exp(-x * x); };
    simplemc::basic_quadrature bq(f, -1.0, 1.0);
    ASSERT_EQ(bq.integration_limits(), (std::array { -1.0, 1.0 }));
    ASSERT_EQ(bq.level(), 1);

    // iterate and check convergence.
    for (int i = 0; i < 15; ++i) {
        bq.next();
    }
    // integral of exp(-x^2) from -1 to 1 is approximately 1.4936...
    ASSERT_NEAR(bq.current(), 1.4936482656248540, 1e-8);
}

// Test basic_quadrature convergence for constant function.
TEST(SimplemcNumeric, BasicQuadratureConstantFunctionConvergence) {
    simplemc::basic_quadrature bq([](double /*x*/) { return 1.0; }, 0.0, 1.0);
    ASSERT_FALSE(bq.is_converged());
    bq.next();
    ASSERT_TRUE(bq.is_converged());
}

// Test trapezoidal quadrature.
TEST(SimplemcNumeric, TrapezoidalQuadrature) {
    auto f = [](double x) { return std::sin(x); };

    // basic convergence test
    auto [res1, err1, n1] = simplemc::trapez_quadrature(f, 0, std::numbers::pi, 1e-8);
    ASSERT_NEAR(res1, 2.0, 2.0 * 1e-8);
    ASSERT_TRUE(err1 < 1e-8 * 2.0);
    fmt::print("Trapezoidal quadrature - Result: {}, N: {}, Error: {}\n", res1, n1, err1);

    // custom min/max iterations
    auto [res2, err2, n2] = simplemc::trapez_quadrature(f, 0, std::numbers::pi, 1e-8, 30, n1 + 2);
    ASSERT_NEAR(res2, 2.0, 2.0 * 1e-8);
    ASSERT_TRUE(err2 < err1);
    ASSERT_EQ(n2, n1 + 2);
    fmt::print("Trapezoidal quadrature - Result: {}, N: {}, Error: {}\n", res2, n2, err2);

    auto [res3, err3, n3] = simplemc::trapez_quadrature(f, 0, std::numbers::pi, 1e-8, n1 - 2);
    ASSERT_TRUE(err3 > err1);
    ASSERT_EQ(n3, n1 - 2);
    fmt::print("Trapezoidal quadrature - Result: {}, N: {}, Error: {}\n", res3, n3, err3);
}

// Test Simpson's quadrature.
TEST(SimplemcNumeric, SimpsonQuadrature) {
    auto f = [](double x) { return std::sin(x); };

    // basic convergence test
    auto [res1, err1, n1] = simplemc::simpson_quadrature(f, 0, std::numbers::pi, 1e-8);
    ASSERT_NEAR(res1, 2.0, 2.0 * 1e-8);
    ASSERT_TRUE(err1 < 1e-8 * 2.0);
    fmt::print("Simpson quadrature - Result: {}, N: {}, Error: {}\n", res1, n1, err1);

    // custom min/max iterations
    auto [res2, err2, n2] = simplemc::simpson_quadrature(f, 0, std::numbers::pi, 1e-8, 30, n1 + 2);
    ASSERT_NEAR(res2, 2.0, 2.0 * 1e-8);
    ASSERT_TRUE(err2 < err1);
    ASSERT_EQ(n2, n1 + 2);
    fmt::print("Simpson quadrature - Result: {}, N: {}, Error: {}\n", res2, n2, err2);

    auto [res3, err3, n3] = simplemc::simpson_quadrature(f, 0, std::numbers::pi, 1e-8, n1 - 2);
    ASSERT_TRUE(err3 > err1);
    ASSERT_EQ(n3, n1 - 2);
    fmt::print("Simpson quadrature - Result: {}, N: {}, Error: {}\n", res3, n3, err3);
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
    fmt::print("Interpolation + quadrature - Result: {}, N: {}, Error: {}\n", res, n, err);
}

// Test quadrature with reversed integration limits (a > b).
TEST(SimplemcNumeric, QuadratureReversedLimits) {
    auto f = [](double x) { return std::sin(x); };
    auto [res, err, n] = simplemc::simpson_quadrature(f, std::numbers::pi, 0.0, 1e-8);
    ASSERT_NEAR(res, -2.0, 2.0 * 1e-8);
}
