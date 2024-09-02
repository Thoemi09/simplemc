#include "../test_utils.hpp"

#include <simplemc/numeric/chebyshev_polynomial.hpp>
#include <simplemc/numeric/hermite_polynomial.hpp>
#include <simplemc/numeric/laguerre_polynomial.hpp>
#include <simplemc/numeric/legendre_polynomial.hpp>
#include <simplemc/numeric/trigonometric_polynomial.hpp>

// Test Legendre polynomials.
TEST(SimplemcNumeric, LegendrePolynomialsFunctionValues) {
    int niter = 10;
    double x = 0.75;
    double tol = 1e-9;
    std::vector<double> boost_l { 1, 0.75, 0.34375, -0.0703125, -0.3500976562, -0.4163818359, -0.2807769775,
        -0.0341835022, 0.1976093054, 0.3103318512 };
    std::vector<double> boost_p { 0, 1, 2.25, 2.71875, 1.7578125, -0.4321289062, -2.822387695, -4.082229614,
        -3.335140228, -0.7228714228 };
    simplemc::legendre_polynomial leg(x);
    for (int i = 0; i < niter; ++i) {
        ASSERT_NEAR(boost_p[i], leg.current_derivative(), tol);
        ASSERT_NEAR(boost_l[i], leg.next(), tol);
        ASSERT_NEAR(boost_p[i], leg.last_derivative(), tol);
    }
}

// Test Chebyshev polynomials.
TEST(SimplemcNumeric, ChebyshevPolynomialsFunctionValues) {
    int niter = 10;
    double x = 0.25;
    double tol = 1e-10;
    std::vector<double> boost_t { 1, 0.25, -0.875, -0.6875, 0.53125, 0.953125, -0.0546875, -0.98046875, -0.435546875,
        0.7626953125 };
    std::vector<double> boost_u { 1, 0.5, -0.75, -0.875, 0.3125, 1.03125, 0.203125, -0.9296875, -0.66796875,
        0.595703125 };
    std::vector<double> boost_tp { 0, 1, 1, -2.25, -3.5, 1.5625, 6.1875, 1.421875, -7.4375, -6.01171875 };
    simplemc::chebyshev_polynomial_first cheb1(x);
    simplemc::chebyshev_polynomial_second cheb2(x);
    for (int i = 0; i < niter; ++i) {
        ASSERT_NEAR(boost_tp[i], cheb1.current_derivative(), tol);
        ASSERT_NEAR(boost_t[i], cheb1.next(), tol);
        ASSERT_NEAR(boost_tp[i], cheb1.last_derivative(), tol);
        ASSERT_NEAR(boost_u[i], cheb2.next(), tol);
    }
}

// Test Laguerre polynomials.
TEST(SimplemcNumeric, LaguerrePolynomialsFunctionValues) {
    int niter = 10;
    double x = 17;
    double tol = 1e-6;
    std::vector<double> boost_l { 1, -16, 111.5, -435.3333333, 1004.708333, -1259.266667, 422.0097222, 838.2230159,
        -578.8142609, -745.0871252 };
    simplemc::laguerre_polynomial lag(x);
    for (int i = 0; i < niter; i++) {
        ASSERT_NEAR(boost_l[i], lag.next(), tol);
    }
}

// Test Hermite polynomials.
TEST(SimplemcNumeric, HermitePolynomialsFunctionValues) {
    int niter = 10;
    double x = -1;
    double tol = 1e-10;
    std::vector<double> boost_h { 1, -2, 2, 4, -20, 8, 184, -464, -1648, 10720 };
    simplemc::hermite_polynomial her(x);
    for (int i = 0; i < niter; ++i) {
        ASSERT_NEAR(boost_h[i], her.next(), tol);
    }
}

// Test cosine polynomials.
TEST(SimplemcNumeric, CosineFunctionValues) {
    int niter = 10;
    double x = 2.345;
    double tol = 1e-10;
    simplemc::cosine cosine(x);
    for (int i = 0; i < niter; ++i) {
        auto j = static_cast<double>(i);
        ASSERT_NEAR(-j * std::sin(j * x), cosine.current_derivative(), tol);
        ASSERT_NEAR(std::cos(j * x), cosine.next(), tol);
        ASSERT_NEAR(-j * std::sin(j * x), cosine.last_derivative(), tol);
    }
}

// Test sine polynomials.
TEST(SimplemcNumeric, SineFunctionValues) {
    int niter = 10;
    double x = 2.345;
    double tol = 1e-10;
    simplemc::sine sine(x);
    for (int i = 0; i < niter; ++i) {
        ASSERT_NEAR(i * std::cos(i * x), sine.current_derivative(), tol);
        ASSERT_NEAR(std::sin(i * x), sine.next(), tol);
        ASSERT_NEAR(i * std::cos(i * x), sine.last_derivative(), tol);
    }
}
