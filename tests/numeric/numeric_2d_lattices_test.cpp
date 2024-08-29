#include "../test_utils.hpp"

#include <simplemc/numeric/bravais_lattice_2d.hpp>

#include <numbers>

using namespace simplemc;
using std::numbers::pi;
using matrix_type = Eigen::Matrix<double, 2, 2>;

// Test 2D square lattice.
TEST(SimplemcNumeric, SquareLattice) {
    double tol = 1e-7;
    auto lat = make_square_lattice(2);
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.0, 0.0, 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0.0, 0.0, pi;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 4.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    ASSERT_EQ(std::string("square_2d"), std::string(lattice_tag_to_string(lat.param().tag)));
    ASSERT_EQ(lattice_tag::square_2d, string_to_lattice_tag("square_2d"));
}

// Test 2D hexagonal lattice.
TEST(SimplemcNumeric, Hexagonal2dLattice) {
    double tol = 1e-7;
    auto lat = make_hexagonal_lattice(2);
    matrix_type exp_real_lat;
    exp_real_lat << 1.0, 1.0, -1.73205081, 1.73205081;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << 3.14159265, 3.14159265, -1.81379936, 1.81379936;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 3.464101615137755;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 11.396437515528111;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    ASSERT_EQ(std::string("hexagonal_2d"), std::string(lattice_tag_to_string(lat.param().tag)));
    ASSERT_EQ(lattice_tag::hexagonal_2d, string_to_lattice_tag("hexagonal_2d"));
}

// Test 2D rectangular lattice.
TEST(SimplemcNumeric, RectangularLattice) {
    double tol = 1e-7;
    auto lat = make_rectangular_lattice(2, 4);
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.0, 0.0, 4.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0.0, 0.0, pi * 0.5;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 8.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi * 0.5;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    ASSERT_EQ(std::string("rectangular_2d"), std::string(lattice_tag_to_string(lat.param().tag)));
    ASSERT_EQ(lattice_tag::rectangular_2d, string_to_lattice_tag("rectangular_2d"));
}

// Test 2D rectangular centered lattice.
TEST(SimplemcNumeric, RectangularCenteredLattice) {
    double tol = 1e-7;
    auto lat = make_rectangular_centered_lattice(2, 4);
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 1.0, 0.0, 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0.0, -pi * 0.5, pi;
    check_range_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);
    double exp_real_vol = 4.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    ASSERT_EQ(std::string("rectangular_centered_2d"), std::string(lattice_tag_to_string(lat.param().tag)));
    ASSERT_EQ(lattice_tag::rectangular_centered_2d, string_to_lattice_tag("rectangular_centered_2d"));
}

// Test 2D oblique lattice.
TEST(SimplemcNumeric, ObliqueLattice) {
    double tol = 1e-7;
    auto lat = make_oblique_lattice(2, 4, 0.5);
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
    ASSERT_EQ(std::string("oblique_2d"), std::string(lattice_tag_to_string(lat.param().tag)));
    ASSERT_EQ(lattice_tag::oblique_2d, string_to_lattice_tag("oblique_2d"));
}
