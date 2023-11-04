/**
 * @file linear_interpolation.cpp
 * @brief Linear interpolation in 1D, 2D and 3D.
 */

#include <simplemc/numeric/linear_interpolation.hpp>

namespace simplemc {

double interpolate_linear(double xd, double f0, double f1) {
    return f0 * (1 - xd) + f1 * xd;
}

double interpolate_bilinear(double xd, double yd, double f00, double f10, double f01, double f11) {
    auto l1 = interpolate_linear(xd, f00, f10);
    auto l2 = interpolate_linear(xd, f01, f11);
    return interpolate_linear(yd, l1, l2);
}

double interpolate_trilinear(double xd, double yd, double zd, double f000, double f100, double f010, double f110,
    double f001, double f101, double f011, double f111) {
    auto b1 = interpolate_bilinear(xd, yd, f000, f100, f010, f110);
    auto b2 = interpolate_bilinear(xd, yd, f001, f101, f011, f111);
    return interpolate_linear(zd, b1, b2);
}

} // namespace simplemc
