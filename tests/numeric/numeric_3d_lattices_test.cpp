// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../test_utils.hpp"

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_3d.hpp>

#include <Eigen/Dense>

#include <numbers>
#include <string>

using namespace simplemc;
using std::numbers::pi;
using matrix_type = Eigen::Matrix<double, 3, 3>;

// Test 3D cubic lattice.
TEST(SimplemcNumeric, CubicLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_cubic_lattice(2);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0.0, 0.0, 0.0, pi, 0.0, 0.0, 0.0, pi;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 8.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi * pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice paramters and tag
    ASSERT_EQ(std::string("cubic_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::cubic_3d, string_to_lattice_tag("cubic_3d"));
}

// Test 3D FCC lattice.
TEST(SimplemcNumeric, FCCLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_fcc_lattice(2);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << -pi, pi, pi, pi, -pi, pi, pi, pi, -pi;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 2.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 124.02510672119922;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("fcc_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::fcc_3d, string_to_lattice_tag("fcc_3d"));
}

// Test 3D BCC lattice.
TEST(SimplemcNumeric, BCCLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_bcc_lattice(2);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << 0, pi, pi, pi, 0, pi, pi, pi, 0;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 4.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 62.01255336059966;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("bcc_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::bcc_3d, string_to_lattice_tag("bcc_3d"));
}

// Test 3D hexagonal lattice.
TEST(SimplemcNumeric, Hexagonal3dLattice) {
    using std::numbers::sqrt3;
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_hexagonal_lattice(2, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 1.0, 1.0, 0.0, -sqrt3, sqrt3, 0.0, 0.0, 0.0, 4.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, pi, 0, -1.81379936, 1.81379936, 0, 0, 0, pi / 2;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 13.856406460551018;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 17.901482187939116;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("hexagonal_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::hexagonal_3d, string_to_lattice_tag("hexagonal_3d"));
}

// Test 3D rhombohedral lattice.
TEST(SimplemcNumeric, RhombohedralLattice) {
    using std::numbers::inv_sqrt3;
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_rhombohedral_lattice(2, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 1, 0, -1, -inv_sqrt3, 1.15470054, -inv_sqrt3, 1.33333333, 1.33333333, 1.33333333;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0, -pi, -1.81379936, 3.62759873, -1.81379936, 1.57079633, 1.57079633, 1.57079633;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 4.618802153517007;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 53.70444656381732;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("rhombohedral_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::rhombohedral_3d, string_to_lattice_tag("rhombohedral_3d"));
}

// Test 3D tetragonal lattice.
TEST(SimplemcNumeric, TetragonalLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_tetragonal_lattice(2, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0.0, 0.0, 0.0, pi, 0.0, 0.0, 0.0, pi / 2;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 16.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * pi * pi / 2;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("tetragonal_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::tetragonal_3d, string_to_lattice_tag("tetragonal_3d"));
}

// Test 3D tetragonal BC lattice.
TEST(SimplemcNumeric, TetragonalBCLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_tetragonal_bc_lattice(2, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << -1, 1, 1, 1, -1, 1, 2, 2, -2;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << 0, pi, pi, pi, 0, pi, pi / 2, pi / 2, 0;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 8.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 31.006276680299827;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("tetragonal_bc_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::tetragonal_bc_3d, string_to_lattice_tag("tetragonal_bc_3d"));
}

// Test 3D orthorhombic lattice.
TEST(SimplemcNumeric, OrthorhombicLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_orthorhombic_lattice(2, 3, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0.0, 0.0, 0.0, 2 * pi / 3, 0.0, 0.0, 0.0, pi / 2;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 24.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi * 2 * pi / 3 * pi / 2;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("orthorhombic_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::orthorhombic_3d, string_to_lattice_tag("orthorhombic_3d"));
}

// Test 3D orthorhombic FC lattice.
TEST(SimplemcNumeric, OrthorhombicFCLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_orthorhombic_fc_lattice(2, 3, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 0, 1, 1, 1.5, 0, 1.5, 2, 2, 0;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << -pi, pi, pi, 2 * pi / 3, -2 * pi / 3, 2 * pi / 3, pi / 2, pi / 2, -pi / 2;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 6.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 41.34170224039976;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("orthorhombic_fc_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::orthorhombic_fc_3d, string_to_lattice_tag("orthorhombic_fc_3d"));
}

// Test 3D orthorhombic BC lattice.
TEST(SimplemcNumeric, OrthorhombicBCLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_orthorhombic_bc_lattice(2, 3, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << -1, 1, 1, 1.5, -1.5, 1.5, 2, 2, -2;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << 0, pi, pi, 2 * pi / 3, 0, 2 * pi / 3, pi / 2, pi / 2, 0;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 12.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 20.67085112019988;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("orthorhombic_bc_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::orthorhombic_bc_3d, string_to_lattice_tag("orthorhombic_bc_3d"));
}

// Test 3D orthorhombic base-centered lattice.
TEST(SimplemcNumeric, OrthorhombicBaseCenteredLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_orthorhombic_base_centered_lattice(2, 3, 4);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 1, 1, 0, -1.5, 1.5, 0, 0, 0, 4;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, pi, 0, -2 * pi / 3, 2 * pi / 3, 0, 0, 0, pi / 2;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 12.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 20.67085112019988;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("orthorhombic_base_centered_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::orthorhombic_base_centered_3d, string_to_lattice_tag("orthorhombic_base_centered_3d"));
}

// Test 3D monoclinic lattice.
TEST(SimplemcNumeric, MonoclinicLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_monoclinic_lattice(2, 3, 4, 0.5);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.0, 3.51033025, 0.0, 3.0, 0.0, 0.0, 0.0, 1.91770215;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0, 0, 0, 2 * pi / 3, 0, -5.75064678, 0, 3.27641354;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 11.506212926500872;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 21.557937005588904;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("monoclinic_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::monoclinic_3d, string_to_lattice_tag("monoclinic_3d"));
}

// Test 3D monoclinic base-centered lattice.
TEST(SimplemcNumeric, MonoclinicBaseCenteredLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_monoclinic_base_centered_lattice(2, 3, 4, 0.5);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 1, 1, 3.51033025, -1.5, 1.5, 0, 0, 0, 1.91770215;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, pi, 0, -2 * pi / 3, 2 * pi / 3, 0, -5.75064678, -5.75064678, 3.27641354;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 5.753106463250437;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 43.11587401117781;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("monoclinic_base_centered_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::monoclinic_base_centered_3d, string_to_lattice_tag("monoclinic_base_centered_3d"));
}

// Test 3D triclinic lattice.
TEST(SimplemcNumeric, TriclinicLattice) {
    double tol = 1e-7;

    // make lattice
    auto [lat, p] = make_triclinic_lattice(2, 3, 4, 0.5, 1, 1.5);

    // real and reciprocal space basis
    matrix_type exp_real_lat;
    exp_real_lat << 2.0, 0.21221161, 2.16120922, 0, 2.99248496, 3.36588394, 0, 0, 5.65685425;
    check_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    matrix_type exp_rec_lat;
    exp_rec_lat << pi, 0, 0, -2.22785554e-01, 2.09965477, 0, -1.06769035, -1.24931525, 1.11072073;
    check_near(make_span(exp_rec_lat), make_span(lat.reciprocal_lattice_vectors()), tol);

    // real and reciprocal space unit cell volumes
    double exp_real_vol = 33.85610252291094;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = 7.326602738000903;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);

    // lattice parameters and tag
    ASSERT_EQ(std::string("triclinic_3d"), lattice_tag_to_string(p.tag));
    ASSERT_EQ(lattice_tag::triclinic_3d, string_to_lattice_tag("triclinic_3d"));
}

// Test exceptions for invalid 3D lattice parameters.
TEST(SimplemcNumeric, Invalid3DLatticeParameters) {
    // invalid lattice constants
    ASSERT_THROW((void)make_cubic_lattice(0), simplemc_exception);
    ASSERT_THROW((void)make_cubic_lattice(-1), simplemc_exception);
    ASSERT_THROW((void)make_orthorhombic_lattice(0, 1, 1), simplemc_exception);
    ASSERT_THROW((void)make_orthorhombic_lattice(1, 0, 1), simplemc_exception);
    ASSERT_THROW((void)make_orthorhombic_lattice(1, 1, 0), simplemc_exception);

    // invalid angles
    ASSERT_THROW((void)make_triclinic_lattice(1, 1, 1, 0, 1, 1), simplemc_exception);
    ASSERT_THROW((void)make_triclinic_lattice(1, 1, 1, 1, 0, 1), simplemc_exception);
    ASSERT_THROW((void)make_triclinic_lattice(1, 1, 1, 1, 1, 0), simplemc_exception);
    ASSERT_THROW((void)make_triclinic_lattice(1, 1, 1, pi + 0.1, 1, 1), simplemc_exception);
}

// Test check_basis_vectors with linearly dependent 3D vectors.
TEST(SimplemcNumeric, CheckBasisVectors3D) {
    matrix_type mat;
    mat << 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0;
    ASSERT_THROW(check_basis_vectors(mat), simplemc_exception);
}

// Test calculate_cell_volume 3D.
TEST(SimplemcNumeric, CalculateCellVolume3D) {
    matrix_type mat;
    mat << 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0;
    ASSERT_DOUBLE_EQ(24.0, calculate_cell_volume(mat));
}

// Test calculate_cell_vertices 3D.
TEST(SimplemcNumeric, CalculateCellVertices3D) {
    matrix_type mat;
    mat << 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0;
    const auto vertices = calculate_cell_vertices(mat);
    // vertex 0: (0,0,0)
    ASSERT_DOUBLE_EQ(0.0, vertices(0, 0));
    ASSERT_DOUBLE_EQ(0.0, vertices(1, 0));
    ASSERT_DOUBLE_EQ(0.0, vertices(2, 0));
    // vertex 1: (2,0,0)
    ASSERT_DOUBLE_EQ(2.0, vertices(0, 1));
    ASSERT_DOUBLE_EQ(0.0, vertices(1, 1));
    ASSERT_DOUBLE_EQ(0.0, vertices(2, 1));
    // vertex 6: (2,3,4)
    ASSERT_DOUBLE_EQ(2.0, vertices(0, 6));
    ASSERT_DOUBLE_EQ(3.0, vertices(1, 6));
    ASSERT_DOUBLE_EQ(4.0, vertices(2, 6));
}
