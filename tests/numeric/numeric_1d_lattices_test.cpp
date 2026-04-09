#include "../test_utils.hpp"

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_1d.hpp>

#include <Eigen/Dense>

#include <numbers>
#include <string>

using namespace simplemc;
using std::numbers::pi;
using matrix_type = Eigen::Matrix<double, 1, 1>;

// Test 1D linear lattice.
TEST(SimplemcNumeric, LinearLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_linear_lattice(2);

    // real and reciprocal space basis
    Eigen::Matrix<double, 1, 1> exp_real_lat;
    exp_real_lat << 2.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    Eigen::Matrix<double, 1, 1> exp_rec_lat;
    exp_rec_lat << pi;
    check_near(make_span(lat.reciprocal_lattice_vectors()), make_span(exp_rec_lat), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 2.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice paramters and tag
    ASSERT_EQ(std::string("linear_1d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::linear_1d, string_to_lattice_tag("linear_1d"));
}

// Test exception for invalid lattice constant.
TEST(SimplemcNumeric, Invalid1DLatticeParameters) {
    ASSERT_THROW((void)make_linear_lattice(0), simplemc_exception);
    ASSERT_THROW((void)make_linear_lattice(-1), simplemc_exception);
}

// Test unspecified lattice tag conversion.
TEST(SimplemcNumeric, UnspecifiedLatticeTag) {
    ASSERT_EQ(std::string("unspecified"), lattice_tag_to_string(lattice_tag::unspecified));
    ASSERT_EQ(lattice_tag::unspecified, string_to_lattice_tag("unspecified"));
}

// Test invalid string to lattice tag conversion.
TEST(SimplemcNumeric, InvalidStringToLatticeTag) {
    ASSERT_THROW((void)string_to_lattice_tag("invalid_lattice"), simplemc_exception);
}

// Test check_basis_vectors with linearly dependent vectors.
TEST(SimplemcNumeric, CheckBasisVectors1D) {
    matrix_type mat;
    mat << 0.0;
    ASSERT_THROW(check_basis_vectors(mat), simplemc_exception);
}

// Test calculate_cell_volume.
TEST(SimplemcNumeric, CalculateCellVolume1D) {
    matrix_type mat;
    mat << 3.0;
    ASSERT_DOUBLE_EQ(3.0, calculate_cell_volume(mat));
}

// Test calculate_cell_vertices.
TEST(SimplemcNumeric, CalculateCellVertices1D) {
    matrix_type mat;
    mat << 2.0;
    const auto vertices = calculate_cell_vertices(mat);
    ASSERT_DOUBLE_EQ(0.0, vertices(0, 0));
    ASSERT_DOUBLE_EQ(2.0, vertices(0, 1));
}
