/**
 * @file numeric_test.cpp
 * @brief Unit tests for simplemc-numeric.
 */

#include "../test_utils.hpp"

#include <simplemc/grids.hpp>
#include <simplemc/numeric.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

#include <numbers>

using namespace simplemc;
using matrix_type = bravais_lattice::matrix_type;
using std::numbers::pi;
const matrix_type unit_matrix_1d = matrix_type::Identity(1, 1);
const matrix_type unit_matrix_2d = matrix_type::Identity(2, 2);
const matrix_type unit_matrix_3d = matrix_type::Identity(3, 3);

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

// Test utilty functions.
TEST(SimplemcNumeric, UtilityFunctions) {
    ASSERT_DOUBLE_EQ(abs_diff(1e-5, 1e-5), 0);
    ASSERT_DOUBLE_EQ(rel_diff(1e-5, 1e-5), 0);
    ASSERT_DOUBLE_EQ(abs_diff(1e-5, 2e-5), 1e-5);
    ASSERT_DOUBLE_EQ(rel_diff(1e-5, 2e-5), 1);
    ASSERT_FALSE(simplemc::isfinite(1.0 / 0.0));
    std::complex<double> c1 { 1.0, std::log(-1) };
    ASSERT_FALSE(isfinite(c1));
    ASSERT_EQ(index_of_subset(4, 9, 1), 4);
    ASSERT_EQ(index_of_subset(4, 9, 2), 3);
    ASSERT_EQ(index_of_subset(4, 9, 3), 3);
    ASSERT_EQ(index_of_subset(4, 9, 6), 1);
    ASSERT_EQ(index_of_subset(4, 9, 8), 0);
    ASSERT_EQ(index_of_subset(4, 9, 9), 0);
    ASSERT_EQ(index_of_subset(7, 9, 1), 7);
    ASSERT_EQ(index_of_subset(7, 9, 2), 6);
    ASSERT_EQ(index_of_subset(7, 9, 3), 6);
    ASSERT_EQ(index_of_subset(7, 9, 4), 5);
    ASSERT_EQ(index_of_subset(7, 9, 7), 2);
    ASSERT_EQ(index_of_subset(1, 9, 1), 1);
    ASSERT_EQ(index_of_subset(1, 9, 2), 0);
    ASSERT_EQ(index_of_subset(1, 9, 3), 0);
    ASSERT_EQ(index_of_subset(1, 9, 4), 0);
    ASSERT_EQ(index_of_subset(1, 9, 9), 0);
}

// Test map_to_interval and map_to_interval_lb functions.
TEST(SimplemcNumeric, MapToInterval) {
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

// Test within_bounds function.
TEST(SimplemcNumeric, WithinBounds) {
    double min_diff = 1e-3;
    double low = 4.2;
    double up = 5.0;
    double val_in1 = 4.5;
    double val_in2 = low + 1e-4;
    double val_in3 = up - 1e-4;
    ASSERT_TRUE(within_bounds(val_in1, low, up, min_diff));
    ASSERT_FALSE(within_bounds(val_in2, low, up, min_diff));
    ASSERT_FALSE(within_bounds(val_in3, low, up, min_diff));
}

// Test conversion between cartesian and polar coordinates.
TEST(SimplemcNumeric, PolarCartesianConversion) {
    vector<3>::type vec { 1.0, 0.0, 0.0 };
    vector<3>::type exp { 1.0, std::numbers::pi / 2, 0.0 };
    check_range_near(cartesian_to_polar(vec), exp);
    vector<2>::type vec2 { std::numbers::sqrt2, std::numbers::pi / 4 };
    vector<2>::type exp2 { 1.0, 1.0 };
    check_range_near(polar_to_cartesian(vec2), exp2);
}

// Test the make_span function for Eigen types.
TEST(SimplemcNumeric, MakeSpan) {
    Eigen::Vector3d v3 = Eigen::Vector3d::Random();
    auto sp_v3 = make_span(v3);
    ASSERT_EQ(sp_v3.size(), v3.size());
    ASSERT_EQ(sp_v3.data(), &v3(0));
    Eigen::VectorXd vn = Eigen::VectorXd::Random(10);
    auto sp_vn = make_span(vn);
    ASSERT_EQ(sp_vn.size(), vn.size());
    ASSERT_EQ(sp_vn.data(), &vn(0));
    Eigen::Matrix3cd m3 = Eigen::Matrix3cd::Random();
    auto sp_m3 = make_span(m3);
    ASSERT_EQ(sp_m3.size(), m3.size());
    ASSERT_EQ(sp_m3.data(), &m3(0, 0));
    Eigen::MatrixXi mn = Eigen::MatrixXi::Random(10, 10);
    auto sp_mn = make_span(mn);
    ASSERT_EQ(sp_mn.size(), mn.size());
    ASSERT_EQ(sp_mn.data(), &mn(0, 0));
}

// Test 1d linear lattice.
TEST(SimplemcNumeric, LinearLattice) {
    double tol = 1e-7;
    auto lat = make_linear_lattice(2);
    auto lat_copy = make_linear_lattice(1);
    matrix_type exp_real_lat(1, 1);
    exp_real_lat << 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(1, 1);
    exp_rec_lat << pi;
    check_range_near(make_span(lat.reciprocal_lattice_vectors()), make_span(exp_rec_lat), tol);
    double exp_real_vol = 2.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_1d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("linear_1d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::linear_1d, string_to_lattice_tag("linear_1d"));
}

// Test 2d square lattice.
TEST(SimplemcNumeric, SquareLattice) {
    double tol = 1e-7;
    auto lat = make_square_lattice(2);
    auto lat_copy = make_square_lattice(1);
    matrix_type exp_real_lat(2, 2);
    exp_real_lat << 2.0, 0.0, 0.0, 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(2, 2);
    exp_rec_lat << pi, 0.0, 0.0, pi;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 4.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_2d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("square_2d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::square_2d, string_to_lattice_tag("square_2d"));
}

// Test 2d hexagonal lattice.
TEST(SimplemcNumeric, Hexagonal2dLattice) {
    double tol = 1e-7;
    auto lat = make_hexagonal_lattice(2);
    auto lat_copy = make_hexagonal_lattice(1);
    matrix_type exp_real_lat(2, 2);
    exp_real_lat << 1.0, 1.0, -1.73205081, 1.73205081;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(2, 2);
    exp_rec_lat << 3.14159265, 3.14159265, -1.81379936, 1.81379936;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 3.464101615137755;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 11.396437515528111;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_2d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("hexagonal_2d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::hexagonal_2d, string_to_lattice_tag("hexagonal_2d"));
}

// Test 2d rectangular lattice.
TEST(SimplemcNumeric, RectangularLattice) {
    double tol = 1e-7;
    auto lat = make_rectangular_lattice(2, 4);
    auto lat_copy = make_rectangular_lattice(1, 1);
    matrix_type exp_real_lat(2, 2);
    exp_real_lat << 2.0, 0.0, 0.0, 4.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(2, 2);
    exp_rec_lat << pi, 0.0, 0.0, pi * 0.5;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 8.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi * 0.5;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_2d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("rectangular_2d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::rectangular_2d, string_to_lattice_tag("rectangular_2d"));
}

// Test 2d rectangular-centered lattice.
TEST(SimplemcNumeric, RectangularCenteredLattice) {
    double tol = 1e-7;
    auto lat = make_rectangular_centered_lattice(2, 4);
    auto lat_copy = make_rectangular_centered_lattice(1, 1);
    matrix_type exp_real_lat(2, 2);
    exp_real_lat << 2.0, 1.0, 0.0, 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(2, 2);
    exp_rec_lat << pi, 0.0, -pi * 0.5, pi;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 4.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_2d), make_span(trans_lat), tol);
    ASSERT_EQ(
        std::string("rectangular_centered_2d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::rectangular_centered_2d, string_to_lattice_tag("rectangular_centered_2d"));
}

// Test 2d oblique lattice.
TEST(SimplemcNumeric, ObliqueLattice) {
    double tol = 1e-7;
    auto lat = make_oblique_lattice(2, 4, 0.5);
    auto lat_copy = make_oblique_lattice(1, 1, 1);
    matrix_type exp_real_lat(2, 2);
    exp_real_lat << 2.0, 3.51033025, 0.0, 1.91770215;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(2, 2);
    exp_rec_lat << 3.14159265, 0.0, -5.75064678, 3.27641354;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 3.835404308833625;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 10.2931567119095;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_2d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("oblique_2d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::oblique_2d, string_to_lattice_tag("oblique_2d"));
}

// Test 3d cubic lattice.
TEST(SimplemcNumeric, CubicLattice) {
    double tol = 1e-7;
    auto lat = make_cubic_lattice(2);
    auto lat_copy = make_cubic_lattice(1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, 0.0, 0.0, 0.0, pi, 0.0, 0.0, 0.0, pi;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 8.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi * pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("cubic_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::cubic_3d, string_to_lattice_tag("cubic_3d"));
}

// Test 3d FCC lattice.
TEST(SimplemcNumeric, FCCLattice) {
    double tol = 1e-7;
    auto lat = make_fcc_lattice(2);
    auto lat_copy = make_fcc_lattice(1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << -pi, pi, pi, pi, -pi, pi, pi, pi, -pi;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 2.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 124.02510672119922;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("fcc_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::fcc_3d, string_to_lattice_tag("fcc_3d"));
}

// Test 3d BCC lattice.
TEST(SimplemcNumeric, BCCLattice) {
    double tol = 1e-7;
    auto lat = make_bcc_lattice(2);
    auto lat_copy = make_bcc_lattice(1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << 0, pi, pi, pi, 0, pi, pi, pi, 0;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 4.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 62.01255336059966;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("bcc_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::bcc_3d, string_to_lattice_tag("bcc_3d"));
}

// Test 3d hexagonal lattice.
TEST(SimplemcNumeric, Hexagonal3dLattice) {
    double tol = 1e-7;
    auto lat = make_hexagonal_lattice(2, 4);
    auto lat_copy = make_hexagonal_lattice(1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 1.0, 1.0, 0.0, -1.73205081, 1.73205081, 0.0, 0.0, 0.0, 4.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, pi, 0, -1.81379936, 1.81379936, 0, 0, 0, pi / 2;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 13.856406460551018;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 17.901482187939116;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("hexagonal_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::hexagonal_3d, string_to_lattice_tag("hexagonal_3d"));
}

// Test 3d rhombohedral lattice.
TEST(SimplemcNumeric, RhombohedralLattice) {
    double tol = 1e-7;
    auto lat = make_rhombohedral_lattice(2, 4);
    auto lat_copy = make_rhombohedral_lattice(1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 1, 0, -1, -0.57735027, 1.15470054, -0.57735027, 1.33333333, 1.33333333, 1.33333333;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, 0, -pi, -1.81379936, 3.62759873, -1.81379936, 1.57079633, 1.57079633, 1.57079633;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 4.618802153517007;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 53.70444656381732;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("rhombohedral_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::rhombohedral_3d, string_to_lattice_tag("rhombohedral_3d"));
}

// Test 3d tetragonal lattice.
TEST(SimplemcNumeric, TetragonalLattice) {
    double tol = 1e-7;
    auto lat = make_tetragonal_lattice(2, 4);
    auto lat_copy = make_tetragonal_lattice(1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, 0.0, 0.0, 0.0, pi, 0.0, 0.0, 0.0, pi / 2;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 16.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi * pi / 2;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("tetragonal_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::tetragonal_3d, string_to_lattice_tag("tetragonal_3d"));
}

// Test 3d tetragonal-BC lattice.
TEST(SimplemcNumeric, TetragonalBCLattice) {
    double tol = 1e-7;
    auto lat = make_tetragonal_bc_lattice(2, 4);
    auto lat_copy = make_tetragonal_bc_lattice(1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << -1, 1, 1, 1, -1, 1, 2, 2, -2;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << 0, pi, pi, pi, 0, pi, pi / 2, pi / 2, 0;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 8.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 31.006276680299827;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("tetragonal_bc_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::tetragonal_bc_3d, string_to_lattice_tag("tetragonal_bc_3d"));
}

// Test 3d orthorhombic lattice.
TEST(SimplemcNumeric, OrthorhombicLattice) {
    double tol = 1e-7;
    auto lat = make_orthorhombic_lattice(2, 3, 4);
    auto lat_copy = make_orthorhombic_lattice(1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, 0.0, 0.0, 0.0, 2 * pi / 3, 0.0, 0.0, 0.0, pi / 2;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 24.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * 2 * pi / 3 * pi / 2;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("orthorhombic_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::orthorhombic_3d, string_to_lattice_tag("orthorhombic_3d"));
}

// Test 3d orthorhombic-FC lattice.
TEST(SimplemcNumeric, OrthorhombicFCLattice) {
    double tol = 1e-7;
    auto lat = make_orthorhombic_fc_lattice(2, 3, 4);
    auto lat_copy = make_orthorhombic_fc_lattice(1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 0, 1, 1, 1.5, 0, 1.5, 2, 2, 0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << -pi, pi, pi, 2 * pi / 3, -2 * pi / 3, 2 * pi / 3, pi / 2, pi / 2, -pi / 2;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 6.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 41.34170224039976;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("orthorhombic_fc_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::orthorhombic_fc_3d, string_to_lattice_tag("orthorhombic_fc_3d"));
}

// Test 3d orthorhombic-BC lattice.
TEST(SimplemcNumeric, OrthorhombicBCLattice) {
    double tol = 1e-7;
    auto lat = make_orthorhombic_bc_lattice(2, 3, 4);
    auto lat_copy = make_orthorhombic_bc_lattice(1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << -1, 1, 1, 1.5, -1.5, 1.5, 2, 2, -2;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << 0, pi, pi, 2 * pi / 3, 0, 2 * pi / 3, pi / 2, pi / 2, 0;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 12.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 20.67085112019988;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("orthorhombic_bc_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::orthorhombic_bc_3d, string_to_lattice_tag("orthorhombic_bc_3d"));
}

// Test 3d orthorhombic-base-centered lattice.
TEST(SimplemcNumeric, OrthorhombicBaseCenteredLattice) {
    double tol = 1e-7;
    auto lat = make_orthorhombic_base_centered_lattice(2, 3, 4);
    auto lat_copy = make_orthorhombic_base_centered_lattice(1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 1, 1, 0, -1.5, 1.5, 0, 0, 0, 4;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, pi, 0, -2 * pi / 3, 2 * pi / 3, 0, 0, 0, pi / 2;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 12.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 20.67085112019988;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("orthorhombic_base_centered_3d"),
        std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::orthorhombic_base_centered_3d,
        string_to_lattice_tag("orthorhombic_base_centered_3d"));
}

// Test 3d monoclinic lattice.
TEST(SimplemcNumeric, Monoclinic_lattice) {
    double tol = 1e-7;
    auto lat = make_monoclinic_lattice(2, 3, 4, 0.5);
    auto lat_copy = make_monoclinic_lattice(1, 1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 2.0, 0.0, 3.51033025, 0.0, 3.0, 0.0, 0.0, 0.0, 1.91770215;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, 0, 0, 0, 2 * pi / 3, 0, -5.75064678, 0, 3.27641354;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 11.506212926500872;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 21.557937005588904;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("monoclinic_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::monoclinic_3d, string_to_lattice_tag("monoclinic_3d"));
}

// Test 3d monoclinic-base-centered lattice.
TEST(SimplemcNumeric, Monoclinic_base_centered_lattice) {
    double tol = 1e-7;
    auto lat = make_monoclinic_base_centered_lattice(2, 3, 4, 0.5);
    auto lat_copy = make_monoclinic_base_centered_lattice(1, 1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 1, 1, 3.51033025, -1.5, 1.5, 0, 0, 0, 1.91770215;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, pi, 0, -2 * pi / 3, 2 * pi / 3, 0, -5.75064678, -5.75064678, 3.27641354;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 5.753106463250437;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 43.11587401117781;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("monoclinic_base_centered_3d"),
        std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::monoclinic_base_centered_3d,
        string_to_lattice_tag("monoclinic_base_centered_3d"));
}

// Test 3d triclinic lattice.
TEST(SimplemcNumeric, Triclinic_lattice) {
    double tol = 1e-7;
    auto lat = make_triclinic_lattice(2, 3, 4, 0.5, 1, 1.5);
    auto lat_copy = make_triclinic_lattice(1, 1, 1, 1, 1, 1);
    matrix_type exp_real_lat(3, 3);
    exp_real_lat << 2.0, 0.21221161, 2.16120922, 0, 2.99248496, 3.36588394, 0, 0, 5.65685425;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat(3, 3);
    exp_rec_lat << pi, 0, 0, -2.22785554e-01, 2.09965477, 0, -1.06769035, -1.24931525, 1.11072073;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 33.85610252291094;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 7.326602738000903;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    matrix_type trans_lat =
        lat.calculate_transformation_matrix(lat.reciprocal_lattice_vectors()) * lat.reciprocal_lattice_vectors();
    check_range_near(make_span(unit_matrix_3d), make_span(trans_lat), tol);
    ASSERT_EQ(std::string("triclinic_3d"), std::string(lattice_tag_to_string(lat.get_lattice_parameters().tag)));
    ASSERT_EQ(bravais_lattice::lattice_tag::triclinic_3d, string_to_lattice_tag("triclinic_3d"));
}

TEST(SimplemcNumeric, LinearInterpolation) {
    double a = 2.0;
    double k = 0.5;
    long nx = 100;
    simplemc::power_grid xgrid(-5.0, 5.0, nx, 2.0);
    Eigen::VectorXd f(nx);
    for (long i = 0; i < nx; ++i) {
        f[i] = line(xgrid.at(i), a, k);
    }
    simplemc::linear_interpolation li(xgrid, f);
    long num = 1233;
    simplemc::linear_grid pts(-5.0, 5.0, num);
    for (long i = 0; i < num; ++i) {
        double x = pts.at(i);
        ASSERT_NEAR(li.interpolate(x), line(x, a, k), 1e-10);
    }
}

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
    simplemc::bilinear_interpolation bi(xgrid, ygrid, make_span(f));
    long num = 37;
    simplemc::linear_grid pts(-1.0, 1.0, num);
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            double x = pts.at(i);
            double y = pts.at(j);
            ASSERT_NEAR(bi.interpolate(x, y), plane(x, y, a, kx, ky), 1e-10);
        }
    }
}

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
    Eigen::Tensor<double, 3, Eigen::RowMajor> f(nx, ny, nz);
    for (long i = 0; i < nx; ++i) {
        for (long j = 0; j < ny; ++j) {
            for (long k = 0; k < nz; ++k) {
                f(i, j, k) = hyperplane(xgrid.at(i), ygrid.at(j), zgrid.at(k), a, kx, ky, kz);
            }
        }
    }
    simplemc::trilinear_interpolation tri(xgrid, ygrid, zgrid, std::span<double>(f.data(), f.size()));
    long num = 37;
    simplemc::linear_grid pts(-2.0, 10.0, num);
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (long k = 0; k < num; ++k) {
                double x = pts.at(i);
                double y = pts.at(j);
                double z = pts.at(k);
                ASSERT_NEAR(tri.interpolate(x, y, z), hyperplane(x, y, z, a, kx, ky, kz), 1e-10);
            }
        }
    }
}
