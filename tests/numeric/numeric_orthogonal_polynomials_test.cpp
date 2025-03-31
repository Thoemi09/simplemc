#include "../test_utils.hpp"

#include <simplemc/numeric/chebyshev_polynomial.hpp>
#include <simplemc/numeric/hermite_polynomial.hpp>
#include <simplemc/numeric/laguerre_polynomial.hpp>
#include <simplemc/numeric/legendre_polynomial.hpp>
#include <simplemc/numeric/orthogonal_polynomial.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// Test orthogonal polynomial concept.
static_assert(simplemc::orthogonal_polynomial<simplemc::legendre_polynomial>);
static_assert(simplemc::orthogonal_polynomial<simplemc::chebyshev_polynomial_first>);
static_assert(simplemc::orthogonal_polynomial<simplemc::chebyshev_polynomial_second>);
static_assert(simplemc::orthogonal_polynomial<simplemc::laguerre_polynomial>);
static_assert(simplemc::orthogonal_polynomial<simplemc::hermite_polynomial>);

// Compare Legendre polynomials with values from boost.
TEST(SimplemcNumeric, LegendrePolynomialsVsBoost) {
    const int max_order = 10;
    const double x = 0.75;
    const double tol = 1e-9;
    std::vector<double> boost_l { 1, 0.75, 0.34375, -0.0703125, -0.3500976562, -0.4163818359, -0.2807769775,
        -0.0341835022, 0.1976093054, 0.3103318512 };
    std::vector<double> boost_p { 0, 1, 2.25, 2.71875, 1.7578125, -0.4321289062, -2.822387695, -4.082229614,
        -3.335140228, -0.7228714228 };
    simplemc::legendre_polynomial pl(x);
    ASSERT_EQ(pl.x(), x);
    ASSERT_EQ(pl.weight(x), 1);
    ASSERT_EQ(pl.domain(), (std::array<double, 2> { -1.0, 1.0 }));
    for (int i = 0; i < max_order; ++i) {
        ASSERT_EQ(pl.order(), i);
        ASSERT_EQ(pl.norm(i), 2.0 / (2 * i + 1));
        ASSERT_NEAR(boost_l[i], pl.current(), tol);
        ASSERT_NEAR(boost_p[i], pl.current_derivative(), tol);
        ASSERT_NEAR(boost_l[i], pl.next(), tol);
        ASSERT_NEAR(boost_p[i], pl.prev_derivative(), tol);
        ASSERT_NEAR(boost_l[i], pl.prev(), tol);
    }
}

// Test the function values and derivatives of Legendre polynomials at the endpoints.
TEST(SimplemcNumeric, LegendrePolynomialsEndpoints) {
    const int max_order = 10;
    const double tol = 1e-10;
    simplemc::legendre_polynomial pl_m1(-1);
    simplemc::legendre_polynomial pl_1(1);
    ASSERT_EQ(pl_m1.x(), -1);
    ASSERT_EQ(pl_1.x(), 1);
    for (int i = 0; i < max_order; ++i) {
        const double p = i * (i + 1) / 2.0;
        const double s = (i % 2 == 0 ? 1 : -1);
        ASSERT_NEAR(p * s * (-1), pl_m1.current_derivative(), tol);
        ASSERT_NEAR(p, pl_1.current_derivative(), tol);
        ASSERT_NEAR(s, pl_m1.next(), tol);
        ASSERT_NEAR(1, pl_1.next(), tol);
    }
}

// Compare Chebyshev polynomials with values from boost.
TEST(SimplemcNumeric, ChebyshevPolynomialsVsBoost) {
    using std::numbers::pi;
    const int max_order = 10;
    const double x = 0.25;
    const double tol = 1e-10;
    std::vector<double> boost_t { 1, 0.25, -0.875, -0.6875, 0.53125, 0.953125, -0.0546875, -0.98046875, -0.435546875,
        0.7626953125 };
    std::vector<double> boost_u { 1, 0.5, -0.75, -0.875, 0.3125, 1.03125, 0.203125, -0.9296875, -0.66796875,
        0.595703125 };
    std::vector<double> boost_tp { 0, 1, 1, -2.25, -3.5, 1.5625, 6.1875, 1.421875, -7.4375, -6.01171875 };
    simplemc::chebyshev_polynomial_first tl(x);
    simplemc::chebyshev_polynomial_second ul(x);
    ASSERT_EQ(tl.x(), x);
    ASSERT_EQ(ul.x(), x);
    ASSERT_EQ(tl.domain(), (std::array<double, 2> { -1.0, 1.0 }));
    ASSERT_EQ(ul.domain(), (std::array<double, 2> { -1.0, 1.0 }));
    for (int i = 0; i < max_order; ++i) {
        ASSERT_EQ(tl.order(), i);
        ASSERT_DOUBLE_EQ(tl.norm(i), (i == 0 ? pi : pi / 2));
        ASSERT_EQ(ul.order(), i);
        ASSERT_DOUBLE_EQ(ul.norm(i), pi / 2);
        ASSERT_NEAR(boost_tp[i], tl.current_derivative(), tol);
        ASSERT_NEAR(boost_t[i], tl.next(), tol);
        ASSERT_NEAR(boost_tp[i], tl.prev_derivative(), tol);
        ASSERT_NEAR(boost_u[i], ul.next(), tol);
    }
}

// Test the function values of Chebyshev polynomials at the endpoints.
TEST(SimplemcNumeric, ChebyshevPolynomialsEndpoints) {
    const int max_order = 10;
    const double tol = 1e-10;
    simplemc::chebyshev_polynomial_first tl_1(1);
    simplemc::chebyshev_polynomial_first tl_m1(-1);
    simplemc::chebyshev_polynomial_second ul_1(1);
    simplemc::chebyshev_polynomial_second ul_m1(-1);
    for (int i = 0; i < max_order; ++i) {
        const double s = (i % 2 == 0 ? 1 : -1);
        const double lp1 = i + 1;
        ASSERT_NEAR(1, tl_1.current(), tol);
        ASSERT_NEAR(s, tl_m1.current(), tol);
        ASSERT_NEAR(lp1, ul_1.current(), tol);
        ASSERT_NEAR(lp1 * s, ul_m1.current(), tol);
        ASSERT_NEAR(1, tl_1.next(), tol);
        ASSERT_NEAR(s, tl_m1.next(), tol);
        ASSERT_NEAR(lp1, ul_1.next(), tol);
        ASSERT_NEAR(lp1 * s, ul_m1.next(), tol);
        ASSERT_NEAR(1, tl_1.prev(), tol);
        ASSERT_NEAR(s, tl_m1.prev(), tol);
        ASSERT_NEAR(lp1, ul_1.prev(), tol);
        ASSERT_NEAR(lp1 * s, ul_m1.prev(), tol);
    }
}

// Test relationships between Chebyshev polynomials and other functions.
TEST(SimplemcNumeric, ChebyshevPolynomialsRelationships) {
    const int max_order = 10;
    const double tol = 1e-10;
    std::vector<double> x_vec { -0.99, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, 0.99 };
    auto exp_tl = [](double x, int n) { return std::cos(n * std::acos(x)); };

    for (auto x : x_vec) {
        simplemc::chebyshev_polynomial_first tl(x);
        simplemc::chebyshev_polynomial_second ul(x);
        ASSERT_EQ(tl.x(), x);
        ASSERT_EQ(ul.x(), x);

        std::vector<double> sums { 0, 0 };
        for (int i = 0; i < max_order; ++i) {
            // weight functions
            ASSERT_DOUBLE_EQ(tl.weight(x), 1 / std::sqrt(1 - x * x));
            ASSERT_DOUBLE_EQ(ul.weight(x), std::sqrt(1 - x * x));

            // T_l(x) = cos(l * arccos(x))
            ASSERT_NEAR(exp_tl(x, i), tl.current(), tol);
            ASSERT_NEAR(exp_tl(x, i), tl.next(), tol);
            ASSERT_NEAR(exp_tl(x, i), tl.prev(), tol);

            // U_l(x) = 2 \sum_{even j >= 0}^{l} T_j(x) - 1 for even l
            // U_l(x) = 2 \sum_{odd j > 0}^{l} T_j(x) for odd l
            sums[i % 2] += tl.prev();
            const auto exp_ul = (i % 2 == 0 ? 2 * sums[0] - 1 : 2 * sums[1]);
            ASSERT_NEAR(exp_ul, ul.current(), tol);
            ASSERT_NEAR(exp_ul, ul.next(), tol);
            ASSERT_NEAR(exp_ul, ul.prev(), tol);

            // T'_l(x) = l U_{l-1}(x)
            ASSERT_NEAR(tl.order() * ul.prev(), tl.current_derivative(), tol);

            // U'_l(x) = ((l + 1) T_{l + 1}(x) - x U_l(x)) / (x^2 - 1)
            ASSERT_NEAR((tl.order() * tl.current() - x * ul.prev()) / (x * x - 1), ul.prev_derivative(), tol);
        }
    }
}

// Compare Laguerre polynomials with values from boost.
TEST(SimplemcNumeric, LaguerrePolynomialsVsBoost) {
    const int max_order = 10;
    const double x = 17;
    const double tol = 1e-6;
    std::vector<double> boost_l { 1, -16, 111.5, -435.3333333, 1004.708333, -1259.266667, 422.0097222, 838.2230159,
        -578.8142609, -745.0871252 };
    simplemc::laguerre_polynomial ll(x);
    ASSERT_EQ(ll.x(), x);
    ASSERT_DOUBLE_EQ(ll.weight(x), std::exp(-x));
    ASSERT_EQ(ll.domain(), (std::array<double, 2> { 0.0, std::numeric_limits<double>::infinity() }));
    for (int i = 0; i < max_order; i++) {
        ASSERT_EQ(ll.order(), i);
        ASSERT_DOUBLE_EQ(ll.norm(i), 1);
        ASSERT_NEAR(boost_l[i], ll.current(), tol);
        ASSERT_NEAR(boost_l[i], ll.next(), tol);
        ASSERT_NEAR(boost_l[i], ll.prev(), tol);
    }
}

// Compare Laguerre polynomials with exact polynomials.
TEST(SimplemcNumeric, LaguerrePolynomialsVsExact) {
    const double tol = 1e-10;
    std::vector<double> x_vec { 0, 0.5, 1, 5, 10, 15 };
    std::vector<std::function<double(double)>> exp_ll, exp_dll;
    exp_ll.emplace_back([]([[maybe_unused]] double x) { return 1; });
    exp_ll.emplace_back([](double x) { return -x + 1; });
    exp_ll.emplace_back([](double x) { return 0.5 * (x * x - 4 * x + 2); });
    exp_ll.emplace_back([](double x) { return -1.0 / 6.0 * (x * x * x - 9 * x * x + 18 * x - 6); });
    exp_ll.emplace_back(
        [](double x) { return 1.0 / 24.0 * (x * x * x * x - 16 * x * x * x + 72 * x * x - 96 * x + 24); });
    exp_dll.emplace_back([]([[maybe_unused]] double x) { return 0; });
    exp_dll.emplace_back([]([[maybe_unused]] double x) { return -1; });
    exp_dll.emplace_back([](double x) { return 0.5 * (2 * x - 4); });
    exp_dll.emplace_back([](double x) { return -1.0 / 6.0 * (3 * x * x - 18 * x + 18); });
    exp_dll.emplace_back([](double x) { return 1.0 / 24.0 * (4 * x * x * x - 48 * x * x + 144 * x - 96); });

    for (auto x : x_vec) {
        simplemc::laguerre_polynomial ll(x);
        ASSERT_DOUBLE_EQ(ll.x(), x);
        ASSERT_DOUBLE_EQ(ll.weight(x), std::exp(-x));
        for (std::size_t i = 0; i < exp_ll.size() - 1; ++i) {
            ASSERT_NEAR(exp_ll[i](x), ll.current(), tol);
            ASSERT_NEAR(exp_dll[i](x), ll.current_derivative(), tol);
            ASSERT_NEAR(exp_ll[i](x), ll.next(), tol);
            ASSERT_NEAR(exp_ll[i](x), ll.prev(), tol);
            ASSERT_NEAR(exp_dll[i](x), ll.prev_derivative(), tol);
        }
    }
}

// Compare Hermite polynomials with values from boost.
TEST(SimplemcNumeric, HermitePolynomialsFunctionValues) {
    const int max_order = 10;
    const double x = -1;
    const double tol = 1e-10;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> boost_h { 1, -2, 2, 4, -20, 8, 184, -464, -1648, 10720 };
    simplemc::hermite_polynomial hl(x);
    ASSERT_EQ(hl.x(), x);
    ASSERT_EQ(hl.weight(x), std::exp(-x * x));
    ASSERT_EQ(hl.domain(), (std::array<double, 2> { -inf, inf }));
    for (int i = 0; i < max_order; ++i) {
        ASSERT_EQ(hl.order(), i);
        ASSERT_DOUBLE_EQ(hl.norm(i), std::sqrt(std::numbers::pi) * std::pow(2, i) * std::tgamma(i + 1));
        ASSERT_NEAR(boost_h[i], hl.next(), tol);
    }
}

// Compare Hermite polynomials with exact polynomials.
TEST(SimplemcNumeric, HermitePolynomialsVsExact) {
    const double tol = 1e-10;
    std::vector<double> x_vec { -15, -10, - 5, -1, -0.5, 0, 0.5, 1, 5, 10, 15 };
    std::vector<std::function<double(double)>> exp_hl, exp_dhl;
    exp_hl.emplace_back([]([[maybe_unused]] double x) { return 1; });
    exp_hl.emplace_back([](double x) { return 2 * x; });
    exp_hl.emplace_back([](double x) { return 4 * x * x - 2; });
    exp_hl.emplace_back([](double x) { return 8 * x * x * x - 12 * x; });
    exp_hl.emplace_back([](double x) { return 16 * x * x * x * x - 48 * x * x + 12; });
    exp_dhl.emplace_back([]([[maybe_unused]] double x) { return 0; });
    exp_dhl.emplace_back([]([[maybe_unused]] double x) { return 2; });
    exp_dhl.emplace_back([](double x) { return 8 * x; });
    exp_dhl.emplace_back([](double x) { return 24 * x * x - 12; });
    exp_dhl.emplace_back([](double x) { return 64 * x * x * x - 96 * x; });

    for (auto x : x_vec) {
        simplemc::hermite_polynomial hl(x);
        ASSERT_DOUBLE_EQ(hl.x(), x);
        ASSERT_DOUBLE_EQ(hl.weight(x), std::exp(-x * x));
        for (std::size_t i = 0; i < exp_hl.size() - 1; ++i) {
            ASSERT_NEAR(exp_hl[i](x), hl.current(), tol);
            ASSERT_NEAR(exp_dhl[i](x), hl.current_derivative(), tol);
            ASSERT_NEAR(exp_hl[i](x), hl.next(), tol);
            ASSERT_NEAR(exp_hl[i](x), hl.prev(), tol);
            ASSERT_NEAR(exp_dhl[i](x), hl.prev_derivative(), tol);
        }
    }
}
