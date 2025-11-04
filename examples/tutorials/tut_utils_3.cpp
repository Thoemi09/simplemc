#include <fmt/base.h>
#include <simplemc/utils.hpp>
#include <simplemc/utils/fmt_complex.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <tuple>
#include <vector>

// Simple custom point class.
struct point {
    double x;
    double y;
    friend std::ostream& operator<<(std::ostream& os, const point& p) { return os << "(" << p.x << ", " << p.y << ")"; }
};

int main() {
    // output a point object
    point p { .x = 1.0, .y = 2.0 };
    std::cout << "Point: " << p << std::endl;

    // print "Hello, World!"
    fmt::println("Hello, World!");

    // print a point object using fmt
    fmt::println("Point: {}", simplemc::to_string(p));
    fmt::println("");

    // lambda to generate n data points on the unit circle in the complex plane
    auto unit_circle_data = [](int n) {
        std::vector<double> angles(n);
        std::vector<std::complex<double>> data(n);
        for (int i = 0; i < n; ++i) {
            angles[i] = i * 2.0 * std::numbers::pi / n;
            data[i] = std::exp(std::complex<double> { 0.0, angles[i] });
        }
        return std::make_tuple(angles, data);
    };

    // generate 10 complex numbers
    int n = 10;
    auto [angles, data] = unit_circle_data(n);

    // print the header of the table
    fmt::println("{:<15}{:<20}{:<20}{}", "Phi(z)", "Re[z]", "Im[z]", "z");

    // print the data of the table
    for (int i = 0; i < n; ++i) {
        fmt::println("{:<15.4f}{:<20.10f}{:<20.10f}{:.4f}", angles[i], data[i].real(), data[i].imag(), data[i]);
    }
}
