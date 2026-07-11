// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric.hpp>

#include <array>
#include <cmath>
#include <vector>

int main() {
    // define the function f(y) = e^y
    auto f = [](double y) { return std::exp(y); };

    // define the domain [c, d] = [2, 5]
    constexpr auto domain_f = std::array<double, 2> { 2.0, 5.0 };

    // linear map y : [-1, 1] -> [2, 5]
    auto y_x = simplemc::linear_map { simplemc::legendre_polynomial::domain(), domain_f };

    // emulate the std::legendre function
    auto legendre = [](int l, double x) {
        auto p_l = simplemc::legendre_polynomial { x };
        for (int i = 0; i < l; ++i) {
            p_l.next();
        }
        return p_l.current();
    };

    // calculate 10 Fourier coefficients
    std::vector<double> coeffs(10);
    for (int k = 0; k < static_cast<int>(coeffs.size()); ++k) {
        // define the integrand g_k(y) = f(y) p_k(x(y)) w(x(y))
        auto g = [&](double y) {
            const auto x_y = y_x.inverse_map(y);
            return f(y) * legendre(k, x_y) * simplemc::legendre_polynomial::weight(x_y);
        };

        // perform the integral I_k = \int_2^5 g_k(y) dy
        auto [i_k, err, n] = simplemc::simpson_quadrature(g, domain_f[0], domain_f[1]);

        // get the coefficient f_k = I_k / (N_k \alpha)
        coeffs[k] = i_k / simplemc::legendre_polynomial::norm(k) / y_x.params().first;
    }

    // lambda to sum the truncated Fourier series S_n(y) = \sum_{l=0}^{n} f_l p_l(x(y))
    auto fourier_sum = [&coeffs, &y_x](int n, double y) {
        auto p_l = simplemc::legendre_polynomial { y_x.inverse_map(y) };
        double sum = 0.0;
        for (int l = 0; l <= n; ++l) {
            sum += coeffs[l] * p_l.next();
        }
        return sum;
    };

    // create the grid at which we want to test our function approximation
    simplemc::linear_grid test_grid { domain_f[0], domain_f[1], 11 };

    // test the function approximation f(y) \approx S_n(y) for n = 3, 5, 7, 10
    fmt::println("{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}", "y", "S_3(y)", "S_5(y)", "S_7(y)", "S_10(y)", "f(y)");
    for (auto y : test_grid) {
        fmt::println("{:<10.1f}{:<20.8g}{:<20.8g}{:<20.8g}{:<20.8g}{:<20.8g}", y, fourier_sum(3, y), fourier_sum(5, y),
            fourier_sum(7, y), fourier_sum(10, y), f(y));
    }
    fmt::println("");
}
