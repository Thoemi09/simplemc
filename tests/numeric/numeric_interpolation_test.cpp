#include "../test_utils.hpp"
#include "./spline.h"

#include <simplemc/grids.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/cubic_spline_interpolation.hpp>
#include <simplemc/numeric/linear_interpolation.hpp>
#include <simplemc/numeric/polynomial_interpolation.hpp>
#include <simplemc/utils/ranges.hpp>

#include <cstddef>
#include <vector>

namespace rgs = simplemc::ranges;

// Construct vector from view.
template <typename R>
auto to_vector(R&& view) { // NOLINT (ranges need not be forwarded)
    auto vec = std::vector<rgs::range_value_t<R>>{};
    vec.reserve(static_cast<std::size_t>(rgs::size(view)));
    for (auto x : view) {
        vec.push_back(x);
    }
    return vec;
}

inline double line(double x, double a = 0.0, double k = 1.0) {
    return a + x * k;
}

inline double plane(double x, double y, double a = 0.0, double kx = 1.0, double ky = 1.0) {
    return a + x * kx + y * ky;
}

inline double hyperplane(
    double x, double y, double z, double a = 0.0, double kx = 1.0, double ky = 1.0, double kz = 1.0) {
    return a + x * kx + y * ky + z * kz;
}

inline double quadratic_poly(double x, double a, double b, double c) {
    return a * x * x + b * x + c;
}

inline double cubic_poly(double x, double a, double b, double c, double d) {
    return a * x * x * x + b * x * x + c * x + d;
}

inline double biquadratic_poly(double x, double y, double a, double b) {
    return a * x * x + b * y * y;
}

inline double tricubic_poly(double x, double y, double z, double a, double b, double c) {
    return a * x * x * x + b * x * y + z * z + c;
}

// Test linear interpolation.
TEST(SimplemcNumeric, LinearInterpolation) {
    double a = 2.0;
    double k = 0.5;
    long nx = 100;
    simplemc::power_grid xgrid(-5.0, 5.0, nx, 2.0);
    Eigen::VectorXd f(nx);
    for (long i = 0; i < nx; ++i) {
        f[i] = line(xgrid.at(i), a, k);
    }
    simplemc::linear_interpolation li(xgrid, simplemc::make_span(f));
    simplemc::linear_interpolation_nd nli(simplemc::nd_grid{xgrid}, simplemc::make_span(f));
    long num = 1233;
    simplemc::linear_grid pts(-5.0, 5.0, num);
    for (long i = 0; i < num; ++i) {
        double x = pts.at(i);
        ASSERT_NEAR(li(x), line(x, a, k), 1e-10);
        ASSERT_NEAR(nli(x), line(x, a, k), 1e-10);
    }
}

// Test bilinear interpolation.
TEST(SimplemcNumeric, BilinearInterpolation) {
    double a = 3.123;
    double kx = 1.2345;
    double ky = -0.89823;
    long nx = 20;
    long ny = 50;
    simplemc::linear_grid xgrid(-1.0, 1.0, nx);
    simplemc::linear_grid ygrid(-1.0, 1.0, ny);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f(nx, ny);
    for (long i = 0; i < nx; ++i) {
        for (long j = 0; j < ny; ++j) {
            f(i, j) = plane(xgrid.at(i), ygrid.at(j), a, kx, ky);
        }
    }
    simplemc::linear_interpolation_nd bi(
        simplemc::nd_grid(xgrid, ygrid), simplemc::make_span(f), simplemc::row_major {});
    long num = 37;
    simplemc::linear_grid pts(-1.0, 1.0, num);
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            double x = pts.at(i);
            double y = pts.at(j);
            ASSERT_NEAR(bi(x, y), plane(x, y, a, kx, ky), 1e-10);
        }
    }
}

// Test trilinear interpolation.
TEST(SimplemcNumeric, TrilinearInterpolation) {
    double a = 3.123;
    double kx = 1.2345;
    double ky = -0.89823;
    double kz = 4.98123;
    long nx = 50;
    long ny = 30;
    long nz = 40;
    simplemc::linear_grid xgrid(-2.0, 10.0, nx);
    simplemc::linear_grid ygrid(-2.0, 10.0, ny);
    simplemc::power_grid zgrid(-2.0, 10.0, nz, 2);
    std::array<long, 3> shape = { nx, ny, nz };
    std::vector<double> f(simplemc::size_from_shape(shape));
    for (long k = 0; k < nz; ++k) {
        for (long j = 0; j < ny; ++j) {
            for (long i = 0; i < nx; ++i) {
                auto idx = simplemc::flat_index(std::array<long, 3> { i, j, k }, shape);
                f[idx] = hyperplane(xgrid.at(i), ygrid.at(j), zgrid.at(k), a, kx, ky, kz);
            }
        }
    }
    simplemc::linear_interpolation_nd tri(simplemc::nd_grid(xgrid, ygrid, zgrid), f);
    long num = 37;
    simplemc::linear_grid pts(-2.0, 10.0, num);
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (long k = 0; k < num; ++k) {
                double x = pts.at(i);
                double y = pts.at(j);
                double z = pts.at(k);
                ASSERT_NEAR(tri(x, y, z), hyperplane(x, y, z, a, kx, ky, kz), 1e-10);
            }
        }
    }
}

// Test polynomial interpolation in 1D.
TEST(SimplemcNumeric, PolynomialInterpolation1D) {
    double a = 3.123;
    double b = -0.8146;
    double c = 1.0;
    double d = 5.91243;
    long nx = 100;
    simplemc::power_grid xgrid(-3.0, 1.0, nx, 3.0);
    std::vector<double> fq(nx);
    std::vector<double> fc(nx);
    for (long i = 0; i < nx; ++i) {
        fq[i] = quadratic_poly(xgrid.at(i), a, b, c);
        fc[i] = cubic_poly(xgrid.at(i), a, b, c, d);
    }
    simplemc::linear_interpolation li(xgrid, fq);
    simplemc::polynomial_interpolation lpi(xgrid, fq, 1);
    simplemc::polynomial_interpolation qi(xgrid, fq, 2);
    simplemc::polynomial_interpolation ci(xgrid, fc, 3);
    simplemc::polynomial_interpolation_nd nqi(simplemc::nd_grid{xgrid}, fq, 2);
    long num = 500;
    simplemc::linear_grid pts(-3.0, 1.0, num);
    for (long i = 0; i < num; ++i) {
        double x = pts.at(i);
        ASSERT_NEAR(qi(x), quadratic_poly(x, a, b, c), 1e-10);
        ASSERT_NEAR(ci(x), cubic_poly(x, a, b, c, d), 1e-10);
        ASSERT_DOUBLE_EQ(nqi(x), qi(x));
        ASSERT_DOUBLE_EQ(li(x), lpi(x));
    }
}

// Test polynomial interpolation in 2D.
TEST(SimplemcNumeric, PolynomialInterpolation2D) {
    double a = 3.123;
    double b = -9.43;
    long nx = 20;
    long ny = 50;
    simplemc::linear_grid xgrid(-1.0, 1.0, nx);
    simplemc::power_grid ygrid(-3.0, 2.0, ny, 2.0);
    std::array<long, 2> shape = { nx, ny };
    std::vector<double> f(nx * ny);
    for (long i = 0; i < nx; ++i) {
        for (long j = 0; j < ny; ++j) {
            auto idx = simplemc::flat_index(std::array<long, 2> { i, j }, shape, simplemc::row_major {});
            f[idx] = biquadratic_poly(xgrid.at(i), ygrid.at(j), a, b);
        }
    }
    simplemc::polynomial_interpolation_nd bi(simplemc::nd_grid(xgrid, ygrid), f, 2, simplemc::row_major {});
    long num = 37;
    simplemc::linear_grid ptsx(-1.0, 1.0, num);
    simplemc::linear_grid ptsy(-3.0, 2.0, num);
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            double x = ptsx.at(i);
            double y = ptsy.at(j);
            ASSERT_NEAR(bi(x, y), biquadratic_poly(x, y, a, b), 1e-10);
        }
    }
}

// Test polynomial interpolation in 3D.
TEST(SimplemcNumeric, PolynomialInterpolation3D) {
    double a = 3.123;
    double b = 1.2345;
    double c = -0.89823;
    long nx = 50;
    long ny = 30;
    long nz = 40;
    simplemc::linear_grid xgrid(-2.0, 10.0, nx);
    simplemc::linear_grid ygrid(-2.0, 10.0, ny);
    simplemc::power_grid zgrid(-2.0, 10.0, nz, 2.0);
    std::array<long, 3> shape = { nx, ny, nz };
    std::vector<double> f(nx * ny * nz);
    for (long k = 0; k < nz; ++k) {
        for (long j = 0; j < ny; ++j) {
            for (long i = 0; i < nx; ++i) {
                auto idx = simplemc::flat_index(std::array<long, 3> { i, j, k }, shape);
                f[idx] = tricubic_poly(xgrid.at(i), ygrid.at(j), zgrid.at(k), a, b, c);
            }
        }
    }
    simplemc::polynomial_interpolation_nd tri(simplemc::nd_grid(xgrid, ygrid, zgrid), f, 3);
    long num = 37;
    simplemc::linear_grid pts(-2.0, 10.0, num);
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (long k = 0; k < num; ++k) {
                double x = pts.at(i);
                double y = pts.at(j);
                double z = pts.at(k);
                ASSERT_NEAR(tri(x, y, z), tricubic_poly(x, y, z, a, b, c), 1e-10);
            }
        }
    }
}

// Test cubic spline interpolation in 1D.
TEST(SimplemcNumeric, CubicSplineInterpolation) {
    double a = 3.123;
    double b = -0.8146;
    double c = 1.0;
    double d = 5.91243;
    long nx = 100;
    simplemc::power_grid xgrid(-3.0, 1.0, nx, 3.0);
    std::vector<double> f(nx);
    for (long i = 0; i < nx; ++i) {
        f[i] = cubic_poly(xgrid.at(i), a, b, c, d);
    }
    double yp_0 = 27 * a - 6 * b + c;
    double yp_n = 3 * a - 2 * b + c;
    simplemc::cubic_spline_interpolation ci(xgrid, f);
    simplemc::cubic_spline_interpolation ci_wf(xgrid, f, yp_0, yp_n);
    auto xvals = to_vector(simplemc::grid_view(xgrid));
    tk::spline tki(xvals, f);
    tk::spline tki_wf(
        xvals, f, tk::spline::cspline, false, tk::spline::first_deriv, yp_0, tk::spline::first_deriv, yp_n);
    long num = 500;
    simplemc::linear_grid pts(-3.0, 1.0, num);
    for (long i = 0; i < num; ++i) {
        double x = pts.at(i);
        ASSERT_NEAR(ci(x), cubic_poly(x, a, b, c, d), 1e-1);
        ASSERT_NEAR(ci(x), tki(x), 1e-5);
        ASSERT_NEAR(ci_wf(x), cubic_poly(x, a, b, c, d), 1e-1);
        ASSERT_NEAR(ci_wf(x), tki_wf(x), 1e-5);
    }
}
