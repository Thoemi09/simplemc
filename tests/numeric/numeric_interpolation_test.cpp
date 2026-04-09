#include "../test_utils.hpp"
#include "./spline.h"

#include <simplemc/grids.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/cubic_spline_interpolation.hpp>
#include <simplemc/numeric/linear_interpolation.hpp>
#include <simplemc/numeric/polynomial_interpolation.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace rgs = simplemc::ranges;

// Construct vector from view.
template <typename R>
auto to_vector(R&& view) { // NOLINT (ranges need not be forwarded)
    auto vec = std::vector<rgs::range_value_t<R>> {};
    vec.reserve(static_cast<std::size_t>(rgs::size(view)));
    for (auto x : view) {
        vec.push_back(x);
    }
    return vec;
}

// Different functions to interpolate.
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
    // parameters for linear function
    double a = 2.0;
    double k = 0.5;

    // interpolation grid
    long nx = 100;
    auto xgrid = simplemc::power_grid { -5.0, 5.0, nx, 2.0 };

    // function values
    auto f = Eigen::VectorXd(nx);
    for (long i = 0; i < nx; ++i) {
        f[i] = line(xgrid.at(i), a, k);
    }

    // interpolation objects
    auto li = simplemc::linear_interpolation { xgrid, simplemc::make_span(f) };
    auto nli = simplemc::linear_interpolation_nd { simplemc::nd_grid { xgrid }, simplemc::make_span(f) };

    // test interpolation
    long num = 1233;
    auto pts = simplemc::linear_grid { -5.0, 5.0, num };
    for (long i = 0; i < num; ++i) {
        double x = pts.at(i);
        ASSERT_NEAR(li(x), line(x, a, k), 1e-10);
        ASSERT_NEAR(nli(x), line(x, a, k), 1e-10);
    }
}

// Test bilinear interpolation.
TEST(SimplemcNumeric, BilinearInterpolation) {
    // parameters for bilinear function
    double a = 3.123;
    double kx = 1.2345;
    double ky = -0.89823;

    // interpolation grid
    long nx = 20;
    long ny = 50;
    auto xgrid = simplemc::linear_grid { -1.0, 1.0, nx };
    auto ygrid = simplemc::linear_grid { -1.0, 1.0, ny };
    auto grid_2d = simplemc::nd_grid { xgrid, ygrid };

    // functions values (row-major order)
    auto f = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(nx, ny);
    for (long i = 0; i < nx; ++i) {
        for (long j = 0; j < ny; ++j) {
            f(i, j) = plane(xgrid.at(i), ygrid.at(j), a, kx, ky);
        }
    }

    // interpolation object
    auto bi = simplemc::linear_interpolation_nd { grid_2d, simplemc::make_span(f), simplemc::row_major {} };

    // test interpolation
    long num = 37;
    auto pts = simplemc::linear_grid { -1.0, 1.0, num };
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
    // parameters for trilinear function
    double a = 3.123;
    double kx = 1.2345;
    double ky = -0.89823;
    double kz = 4.98123;

    // interpolation grid
    long nx = 50;
    long ny = 30;
    long nz = 40;
    auto xgrid = simplemc::linear_grid { -2.0, 10.0, nx };
    auto ygrid = simplemc::linear_grid { -2.0, 10.0, ny };
    auto zgrid = simplemc::power_grid { -2.0, 10.0, nz, 2 };
    auto grid_3d = simplemc::nd_grid { xgrid, ygrid, zgrid };

    // function values (column-major order)
    auto shape = std::array<long, 3> { nx, ny, nz };
    auto f = std::vector<double>(simplemc::size_from_shape(shape));
    for (long k = 0; k < nz; ++k) {
        for (long j = 0; j < ny; ++j) {
            for (long i = 0; i < nx; ++i) {
                auto idx = simplemc::flat_index(std::array<long, 3> { i, j, k }, shape);
                f[idx] = hyperplane(xgrid.at(i), ygrid.at(j), zgrid.at(k), a, kx, ky, kz);
            }
        }
    }

    // interpolation object
    auto tri = simplemc::linear_interpolation_nd { grid_3d, f };

    // test interpolation
    long num = 37;
    auto pts = simplemc::linear_grid { -2.0, 10.0, num };
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
    // parameters for polynomials
    double a = 3.123;
    double b = -0.8146;
    double c = 1.0;
    double d = 5.91243;

    // interpolation grid
    long nx = 100;
    auto xgrid = simplemc::power_grid { -3.0, 1.0, nx, 3.0 };

    // function values
    auto fq = std::vector<double>(nx);
    auto fc = std::vector<double>(nx);
    for (long i = 0; i < nx; ++i) {
        fq[i] = quadratic_poly(xgrid.at(i), a, b, c);
        fc[i] = cubic_poly(xgrid.at(i), a, b, c, d);
    }

    // interpolation objects
    auto li = simplemc::linear_interpolation { xgrid, fq };
    auto lpi = simplemc::polynomial_interpolation { xgrid, fq, 1 };
    auto qi = simplemc::polynomial_interpolation { xgrid, fq, 2 };
    auto ci = simplemc::polynomial_interpolation { xgrid, fc, 3 };
    auto nqi = simplemc::polynomial_interpolation_nd { simplemc::nd_grid { xgrid }, fq, 2 };

    // test interpolation
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
    // parameter for polynomials
    double a = 3.123;
    double b = -9.43;

    // interpolation grids
    long nx = 20;
    long ny = 50;
    auto xgrid = simplemc::linear_grid { -1.0, 1.0, nx };
    auto ygrid = simplemc::power_grid { -3.0, 2.0, ny, 2.0 };
    auto grid_2d = simplemc::nd_grid { xgrid, ygrid };

    // function values (row-major order)
    auto shape = std::array<long, 2> { nx, ny };
    auto f = std::vector<double>(nx * ny);
    for (long i = 0; i < nx; ++i) {
        for (long j = 0; j < ny; ++j) {
            auto idx = simplemc::flat_index(std::array<long, 2> { i, j }, shape, simplemc::row_major {});
            f[idx] = biquadratic_poly(xgrid.at(i), ygrid.at(j), a, b);
        }
    }

    // interpolation object
    auto bi = simplemc::polynomial_interpolation_nd { grid_2d, f, 2, simplemc::row_major {} };
    long num = 37;
    auto ptsx = simplemc::linear_grid { -1.0, 1.0, num };
    auto ptsy = simplemc::linear_grid { -3.0, 2.0, num };
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
    // parameters for polynomials
    double a = 3.123;
    double b = 1.2345;
    double c = -0.89823;

    // interpolation grid
    long nx = 50;
    long ny = 30;
    long nz = 40;
    auto xgrid = simplemc::linear_grid { -2.0, 10.0, nx };
    auto ygrid = simplemc::linear_grid { -2.0, 10.0, ny };
    auto zgrid = simplemc::power_grid { -2.0, 10.0, nz, 2.0 };
    auto grid_3d = simplemc::nd_grid { xgrid, ygrid, zgrid };

    // function values (column-major order)
    auto shape = std::array<long, 3> { nx, ny, nz };
    auto f = std::vector<double>(nx * ny * nz);
    for (long k = 0; k < nz; ++k) {
        for (long j = 0; j < ny; ++j) {
            for (long i = 0; i < nx; ++i) {
                auto idx = simplemc::flat_index(std::array<long, 3> { i, j, k }, shape);
                f[idx] = tricubic_poly(xgrid.at(i), ygrid.at(j), zgrid.at(k), a, b, c);
            }
        }
    }

    // interpolation object
    auto tri = simplemc::polynomial_interpolation_nd { grid_3d, f, 3 };

    // test interpolation
    long num = 37;
    auto pts = simplemc::linear_grid { -2.0, 10.0, num };
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
    // parameter for cubic polynomial
    double a = 3.123;
    double b = -0.8146;
    double c = 1.0;
    double d = 5.91243;

    // interpolation grid
    long nx = 100;
    auto xgrid = simplemc::power_grid { -3.0, 1.0, nx, 3.0 };

    // function values
    std::vector<double> f(nx);
    for (long i = 0; i < nx; ++i) {
        f[i] = cubic_poly(xgrid.at(i), a, b, c, d);
    }

    // interpolation objects
    double gamma_0 = 27 * a - 6 * b + c;
    double gamma_m1 = 3 * a - 2 * b + c;
    auto ci = simplemc::cubic_spline_interpolation { xgrid, f };
    auto ci_wf = simplemc::cubic_spline_interpolation { xgrid, f, gamma_0, gamma_m1 };

    // reference spline
    auto xvals = to_vector(simplemc::grid_view(xgrid));
    auto tki = tk::spline { xvals, f };
    auto tki_wf = tk::spline { xvals, f, tk::spline::cspline, false, tk::spline::first_deriv, gamma_0,
        tk::spline::first_deriv, gamma_m1 };

    // test interpolation
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

// Test linear_interpolation accessors.
TEST(SimplemcNumeric, LinearInterpolationAccessors) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double> { 0.0, 0.25, 0.5, 0.75, 1.0 };
    auto li = simplemc::linear_interpolation { xgrid, f };
    ASSERT_EQ(li.dim(), 1);
    check_equal(li.grid(), xgrid);
    check_equal(li.function_values(), f);
}

// Test linear_interpolation_nd accessors.
TEST(SimplemcNumeric, LinearInterpolationNdAccessors) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 3 };
    auto ygrid = simplemc::linear_grid { 0.0, 2.0, 4 };
    auto grid_2d = simplemc::nd_grid { xgrid, ygrid };
    auto f = std::vector<double>(12, 1.0);
    auto bi = simplemc::linear_interpolation_nd { grid_2d, f };
    ASSERT_EQ(bi.dim(), 2);
    check_equal(std::get<0>(bi.grid().grids()), xgrid);
    check_equal(std::get<1>(bi.grid().grids()), ygrid);
    check_equal(bi.function_values(), f);
}

// Test polynomial_interpolation accessors.
TEST(SimplemcNumeric, PolynomialInterpolationAccessors) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 10 };
    auto f = std::vector<double>(10, 1.0);
    auto pi = simplemc::polynomial_interpolation { xgrid, f, 3 };
    ASSERT_EQ(pi.dim(), 1);
    ASSERT_EQ(pi.degree(), 3);
    check_equal(pi.grid(), xgrid);
    check_equal(pi.function_values(), f);
}

// Test polynomial_interpolation_nd accessors.
TEST(SimplemcNumeric, PolynomialInterpolationNdAccessors) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto grid_2d = simplemc::nd_grid { xgrid, xgrid };
    auto f = std::vector<double>(25, 1.0);
    auto pi = simplemc::polynomial_interpolation_nd { grid_2d, f, 2 };
    ASSERT_EQ(pi.dim(), 2);
    ASSERT_EQ(pi.degree(), 2);
    check_equal(std::get<0>(pi.grid().grids()), xgrid);
    check_equal(std::get<1>(pi.grid().grids()), xgrid);
    check_equal(pi.function_values(), f);
}

// Test cubic_spline_interpolation accessors.
TEST(SimplemcNumeric, CubicSplineInterpolationAccessors) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 10 };
    auto f = std::vector<double>(10, 1.0);
    auto ci = simplemc::cubic_spline_interpolation { xgrid, f };
    ASSERT_EQ(ci.dim(), 1);
    check_equal(ci.grid(), xgrid);
    check_equal(ci.function_values(), f);
}

// Test linear_interpolation exception on size mismatch.
TEST(SimplemcNumeric, LinearInterpolationSizeMismatch) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double>(10, 1.0);
    ASSERT_THROW(simplemc::linear_interpolation(xgrid, f), simplemc::simplemc_exception);
}

// Test linear_interpolation_nd exception on size mismatch.
TEST(SimplemcNumeric, LinearInterpolationNdSizeMismatch) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto grid_2d = simplemc::nd_grid { xgrid, xgrid };
    auto f = std::vector<double>(10, 1.0);
    ASSERT_THROW(simplemc::linear_interpolation_nd(grid_2d, f), simplemc::simplemc_exception);
}

// Test polynomial_interpolation exception on size mismatch.
TEST(SimplemcNumeric, PolynomialInterpolationSizeMismatch) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double>(10, 1.0);
    ASSERT_THROW(simplemc::polynomial_interpolation(xgrid, f, 2), simplemc::simplemc_exception);
}

// Test polynomial_interpolation exception on degree too high.
TEST(SimplemcNumeric, PolynomialInterpolationDegreeTooHigh) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double>(5, 1.0);
    ASSERT_THROW(simplemc::polynomial_interpolation(xgrid, f, 5), simplemc::simplemc_exception);
}

// Test cubic_spline_interpolation exception on size mismatch.
TEST(SimplemcNumeric, CubicSplineInterpolationSizeMismatch) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double>(10, 1.0);
    ASSERT_THROW(simplemc::cubic_spline_interpolation(xgrid, f), simplemc::simplemc_exception);
}

// Test cubic_spline_interpolation with custom matrix/vector - size mismatch.
TEST(SimplemcNumeric, CubicSplineInterpolationMatrixSizeMismatch) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double>(5, 1.0);
    auto mat = Eigen::MatrixXd::Identity(3, 3);
    auto vec = Eigen::VectorXd::Zero(5);
    ASSERT_THROW(simplemc::cubic_spline_interpolation(xgrid, f, mat, vec), simplemc::simplemc_exception);
}

// Test cubic_spline_interpolation with custom matrix/vector - vector size mismatch.
TEST(SimplemcNumeric, CubicSplineInterpolationVectorSizeMismatch) {
    auto xgrid = simplemc::linear_grid { 0.0, 1.0, 5 };
    auto f = std::vector<double>(5, 1.0);
    auto mat = Eigen::MatrixXd::Identity(5, 5);
    auto vec = Eigen::VectorXd::Zero(3);
    ASSERT_THROW(simplemc::cubic_spline_interpolation(xgrid, f, mat, vec), simplemc::simplemc_exception);
}

// Test interpolation at exact grid points.
TEST(SimplemcNumeric, InterpolationAtGridPoints) {
    auto xgrid = simplemc::linear_grid { 0.0, 4.0, 5 };
    auto f = std::vector<double> { 0.0, 1.0, 4.0, 9.0, 16.0 };

    auto li = simplemc::linear_interpolation { xgrid, f };
    auto pi = simplemc::polynomial_interpolation { xgrid, f, 2 };
    auto ci = simplemc::cubic_spline_interpolation { xgrid, f };

    // test at exact grid points
    for (long i = 0; i < 5; ++i) {
        double x = xgrid.at(i);
        ASSERT_NEAR(li(x), f[i], 1e-10);
        ASSERT_NEAR(pi(x), f[i], 1e-10);
        ASSERT_NEAR(ci(x), f[i], 1e-10);
    }
}

// Test interpolation at domain boundaries.
TEST(SimplemcNumeric, InterpolationAtBoundaries) {
    auto xgrid = simplemc::linear_grid { -1.0, 1.0, 11 };
    auto f = std::vector<double>(11);
    for (long i = 0; i < 11; ++i) {
        f[i] = xgrid.at(i) * xgrid.at(i);
    }

    auto li = simplemc::linear_interpolation { xgrid, f };
    auto pi = simplemc::polynomial_interpolation { xgrid, f, 2 };
    auto ci = simplemc::cubic_spline_interpolation { xgrid, f };

    // test at boundaries
    ASSERT_NEAR(li(-1.0), 1.0, 1e-10);
    ASSERT_NEAR(li(1.0), 1.0, 1e-10);
    ASSERT_NEAR(pi(-1.0), 1.0, 1e-10);
    ASSERT_NEAR(pi(1.0), 1.0, 1e-10);
    ASSERT_NEAR(ci(-1.0), 1.0, 1e-10);
    ASSERT_NEAR(ci(1.0), 1.0, 1e-10);
}

// Test cubic spline with custom_grid.
TEST(SimplemcNumeric, CubicSplineInterpolationCustomGrid) {
    auto xvals = std::vector<double> { 0.0, 0.5, 1.0, 1.5, 2.0 };
    auto xgrid = simplemc::custom_grid { xvals };
    auto f = std::vector<double>(5);
    for (long i = 0; i < 5; ++i) {
        f[i] = xvals[i] * xvals[i] * xvals[i]; // f(x) = x^3
    }

    auto ci = simplemc::cubic_spline_interpolation { xgrid, f };

    // cubic spline should exactly interpolate cubic polynomial at grid points
    for (long i = 0; i < 5; ++i) {
        ASSERT_NEAR(ci(xvals[i]), f[i], 1e-10);
    }
}

// Test that linear_interpolation static methods are constexpr.
TEST(SimplemcNumeric, LinearInterpolationConstexpr) {
    using namespace simplemc;
    static_assert(linear_interpolation<linear_grid>::dim() == 1);
    static_assert(linear_interpolation_nd<nd_grid<linear_grid, linear_grid>>::dim() == 2);
    constexpr auto test_constexpr = [] {
        auto xgrid = linear_grid { 0.0, 1.0, 3 };
        std::array<double, 3> f_arr = { 0.0, 0.5, 1.0 };
        auto li = linear_interpolation<linear_grid> { xgrid, std::span<double> { f_arr } };
        return li(0.25);
    };
    constexpr double result = test_constexpr();
    ASSERT_DOUBLE_EQ(result, 0.25);
}

// Test cubic spline interpolation with custom matrix and vector (natural BCs reconstructed).
TEST(SimplemcNumeric, CubicSplineInterpolationCustomMatrixVector) {
    // create a simple grid and function values
    const long nx = 5;
    auto xgrid = simplemc::linear_grid { 0.0, 4.0, nx };
    auto f = std::vector<double>(nx);
    for (long i = 0; i < nx; ++i) {
        f[i] = xgrid.at(i) * xgrid.at(i); // f(x) = x^2
    }

    // create natural boundary condition spline for reference
    auto ci_natural = simplemc::cubic_spline_interpolation { xgrid, f };

    // manually construct the natural BC spline matrix and vector
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nx, nx);
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(nx);

    // natural BCs: gamma_0 = 0, gamma_{n-1} = 0
    mat(0, 0) = 1.0;
    vec(0) = 0.0;
    mat(nx - 1, nx - 1) = 1.0;
    vec(nx - 1) = 0.0;

    // interior equations (tridiagonal structure)
    for (long i = 1; i < nx - 1; ++i) {
        const double xim1 = xgrid.at(i - 1);
        const double xi = xgrid.at(i);
        const double xip1 = xgrid.at(i + 1);
        mat(i, i - 1) = xi - xim1;
        mat(i, i) = 2.0 * (xip1 - xim1);
        mat(i, i + 1) = xip1 - xi;
        vec(i) = 6.0 * ((f[i + 1] - f[i]) / (xip1 - xi) - (f[i] - f[i - 1]) / (xi - xim1));
    }

    // create spline with custom matrix/vector
    auto ci_custom = simplemc::cubic_spline_interpolation { xgrid, f, mat, vec };

    // both should give the same results
    const long num = 20;
    auto pts = simplemc::linear_grid { 0.0, 4.0, num };
    for (long i = 0; i < num; ++i) {
        double x = pts.at(i);
        ASSERT_NEAR(ci_custom(x), ci_natural(x), 1e-10);
    }
}
