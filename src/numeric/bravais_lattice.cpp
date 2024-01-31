/**
 * @file bravais_lattice.cpp
 * @brief Bravais lattices in 1d, 2d and 3d.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cmath>
#include <numbers>
#include <utility>

namespace simplemc {

void bravais_lattice::check_lattice_vector_dims(const matrix_type& mat) {
    if (mat.rows() != mat.cols() || mat.rows() == 0 || mat.rows() > 3) {
        throw simplemc_exception(
            "Lattice vector matrix dimensions not supported", "bravais_lattice::check_lattice_vector_dims");
    }
}

double bravais_lattice::calculate_cell_volume(const matrix_type& mat) {
    bravais_lattice::check_lattice_vector_dims(mat);
    return std::abs(mat.determinant());
}

bravais_lattice::matrix_type bravais_lattice::calculate_reciprocal_lattice(const matrix_type& mat) {
    using std::numbers::pi;
    bravais_lattice::check_lattice_vector_dims(mat);
    matrix_type B = vector_type::Constant(mat.rows(), 2 * pi).asDiagonal();
    return mat.transpose().fullPivLu().solve(B);
}

bravais_lattice::matrix_type bravais_lattice::calculate_cell_vertices(const matrix_type& mat) {
    bravais_lattice::check_lattice_vector_dims(mat);
    if (mat.rows() == 1) {
        matrix_type res = matrix_type::Zero(1, 2);
        res(0, 1) = mat(0, 0);
        return res;
    } else if (mat.rows() == 2) {
        matrix_type res = matrix_type::Zero(2, 4);
        res.col(1) = mat.col(0);
        res.col(2) = mat.col(0) + mat.col(1);
        res.col(3) = mat.col(1);
        return res;
    } else {
        matrix_type res = matrix_type::Zero(3, 8);
        res.col(1) = mat.col(0);
        res.col(2) = mat.col(0) + mat.col(1);
        res.col(3) = mat.col(1);
        res.col(4) = mat.col(2);
        res.col(5) = mat.col(2) + mat.col(0);
        res.col(6) = mat.col(2) + mat.col(0) + mat.col(1);
        res.col(7) = mat.col(2) + mat.col(1);
        return res;
    }
}

bravais_lattice::matrix_type bravais_lattice::calculate_transformation_matrix(const matrix_type& mat) {
    bravais_lattice::check_lattice_vector_dims(mat);
    if (mat.rows() == 1) {
        matrix_type res(1, 1);
        res(0, 0) = 1.0 / mat(0, 0);
        return res;
    } else if (mat.rows() == 2) {
        matrix_type a = matrix_type::Zero(4, 4);
        a.block<1, 2>(0, 0) = mat.col(0);
        a.block<1, 2>(1, 0) = mat.col(1);
        a.block<1, 2>(2, 2) = mat.col(0);
        a.block<1, 2>(3, 2) = mat.col(1);
        vector_type b(4);
        b << 1, 0, 0, 1;
        vector_type x = a.fullPivLu().solve(b);
        matrix_type res(2, 2);
        res << x(0), x(1), x(2), x(3);
        return res;
    } else {
        matrix_type a = matrix_type::Zero(9, 9);
        a.block<1, 3>(0, 0) = mat.col(0);
        a.block<1, 3>(1, 0) = mat.col(1);
        a.block<1, 3>(2, 0) = mat.col(2);
        a.block<1, 3>(3, 3) = mat.col(0);
        a.block<1, 3>(4, 3) = mat.col(1);
        a.block<1, 3>(5, 3) = mat.col(2);
        a.block<1, 3>(6, 6) = mat.col(0);
        a.block<1, 3>(7, 6) = mat.col(1);
        a.block<1, 3>(8, 6) = mat.col(2);
        vector_type b(9);
        b << 1, 0, 0, 0, 1, 0, 0, 0, 1;
        vector_type x = a.fullPivLu().solve(b);
        matrix_type res(3, 3);
        res << x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8);
        return res;
    }
}

bravais_lattice::bravais_lattice(matrix_type mat, const params& p) :
    params_(p),
    real_lat_(std::move(mat)),
    rec_lat_(calculate_reciprocal_lattice(real_lat_)),
    real_vol_(calculate_cell_volume(real_lat_)),
    rec_vol_(calculate_cell_volume(rec_lat_)) {}

void bravais_lattice::set_lattice_vectors(matrix_type mat) {
    rec_lat_ = calculate_reciprocal_lattice(mat);
    real_vol_ = calculate_cell_volume(mat);
    rec_vol_ = calculate_cell_volume(mat);
    real_lat_ = std::move(mat);
}

void bravais_lattice::set_lattice_parameters(const params& p) {
    params_ = p;
}

std::string lattice_tag_to_string(const bravais_lattice::lattice_tag& tag) {
    if (tag == bravais_lattice::lattice_tag::linear_1d) {
        return "linear_1d";
    } else if (tag == bravais_lattice::lattice_tag::square_2d) {
        return "square_2d";
    } else if (tag == bravais_lattice::lattice_tag::hexagonal_2d) {
        return "hexagonal_2d";
    } else if (tag == bravais_lattice::lattice_tag::rectangular_2d) {
        return "rectangular_2d";
    } else if (tag == bravais_lattice::lattice_tag::rectangular_centered_2d) {
        return "rectangular_centered_2d";
    } else if (tag == bravais_lattice::lattice_tag::oblique_2d) {
        return "oblique_2d";
    } else if (tag == bravais_lattice::lattice_tag::cubic_3d) {
        return "cubic_3d";
    } else if (tag == bravais_lattice::lattice_tag::fcc_3d) {
        return "fcc_3d";
    } else if (tag == bravais_lattice::lattice_tag::bcc_3d) {
        return "bcc_3d";
    } else if (tag == bravais_lattice::lattice_tag::hexagonal_3d) {
        return "hexagonal_3d";
    } else if (tag == bravais_lattice::lattice_tag::rhombohedral_3d) {
        return "rhombohedral_3d";
    } else if (tag == bravais_lattice::lattice_tag::tetragonal_3d) {
        return "tetragonal_3d";
    } else if (tag == bravais_lattice::lattice_tag::tetragonal_bc_3d) {
        return "tetragonal_bc_3d";
    } else if (tag == bravais_lattice::lattice_tag::orthorhombic_3d) {
        return "orthorhombic_3d";
    } else if (tag == bravais_lattice::lattice_tag::orthorhombic_fc_3d) {
        return "orthorhombic_fc_3d";
    } else if (tag == bravais_lattice::lattice_tag::orthorhombic_bc_3d) {
        return "orthorhombic_bc_3d";
    } else if (tag == bravais_lattice::lattice_tag::orthorhombic_base_centered_3d) {
        return "orthorhombic_base_centered_3d";
    } else if (tag == bravais_lattice::lattice_tag::monoclinic_3d) {
        return "monoclinic_3d";
    } else if (tag == bravais_lattice::lattice_tag::monoclinic_base_centered_3d) {
        return "monoclinic_base_centered_3d";
    } else if (tag == bravais_lattice::lattice_tag::triclinic_3d) {
        return "triclinic_3d";
    } else if (tag == bravais_lattice::lattice_tag::unspecified) {
        return "unspecified";
    } else {
        throw simplemc_exception("Wrong lattice tag", "lattice_tag_to_string");
    }
}

bravais_lattice::lattice_tag string_to_lattice_tag(const std::string& str) {
    if (str == "linear_1d") {
        return bravais_lattice::lattice_tag::linear_1d;
    } else if (str == "square_2d") {
        return bravais_lattice::lattice_tag::square_2d;
    } else if (str == "hexagonal_2d") {
        return bravais_lattice::lattice_tag::hexagonal_2d;
    } else if (str == "rectangular_2d") {
        return bravais_lattice::lattice_tag::rectangular_2d;
    } else if (str == "rectangular_centered_2d") {
        return bravais_lattice::lattice_tag::rectangular_centered_2d;
    } else if (str == "oblique_2d") {
        return bravais_lattice::lattice_tag::oblique_2d;
    } else if (str == "cubic_3d") {
        return bravais_lattice::lattice_tag::cubic_3d;
    } else if (str == "fcc_3d") {
        return bravais_lattice::lattice_tag::fcc_3d;
    } else if (str == "bcc_3d") {
        return bravais_lattice::lattice_tag::bcc_3d;
    } else if (str == "hexagonal_3d") {
        return bravais_lattice::lattice_tag::hexagonal_3d;
    } else if (str == "rhombohedral_3d") {
        return bravais_lattice::lattice_tag::rhombohedral_3d;
    } else if (str == "tetragonal_3d") {
        return bravais_lattice::lattice_tag::tetragonal_3d;
    } else if (str == "tetragonal_bc_3d") {
        return bravais_lattice::lattice_tag::tetragonal_bc_3d;
    } else if (str == "orthorhombic_3d") {
        return bravais_lattice::lattice_tag::orthorhombic_3d;
    } else if (str == "orthorhombic_fc_3d") {
        return bravais_lattice::lattice_tag::orthorhombic_fc_3d;
    } else if (str == "orthorhombic_bc_3d") {
        return bravais_lattice::lattice_tag::orthorhombic_bc_3d;
    } else if (str == "orthorhombic_base_centered_3d") {
        return bravais_lattice::lattice_tag::orthorhombic_base_centered_3d;
    } else if (str == "monoclinic_3d") {
        return bravais_lattice::lattice_tag::monoclinic_3d;
    } else if (str == "monoclinic_base_centered_3d") {
        return bravais_lattice::lattice_tag::monoclinic_base_centered_3d;
    } else if (str == "triclinic_3d") {
        return bravais_lattice::lattice_tag::triclinic_3d;
    } else if (str == "unspecified") {
        return bravais_lattice::lattice_tag::unspecified;
    } else {
        throw simplemc_exception(fmt::format("Wrong lattice string: {}", str), "string_to_lattice_tag");
    }
}

void check_1d_lattice_parameters(const bravais_lattice::params& p) {
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0", "check_1d_lattice_parameters");
    }
}

bravais_lattice make_linear_lattice(double a) {
    bravais_lattice::matrix_type mat(1, 1);
    mat << a;
    bravais_lattice::params p;
    p.a = a;
    p.tag = bravais_lattice::lattice_tag::linear_1d;
    check_1d_lattice_parameters(p);
    return { mat, p };
}

void check_2d_lattice_parameters(const bravais_lattice::params& p) {
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0", "check_2d_lattice_parameters");
    }
    if (p.b <= 0.0) {
        throw simplemc_exception("b <= 0", "check_2d_lattice_parameters");
    }
    if (simplemc::abs_diff(std::sin(p.gamma), 0.0) < 1e-14) {
        throw simplemc_exception("gamma is a multiple of pi", "check_2d_lattice_parameters");
    }
}

bravais_lattice make_square_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice::matrix_type mat(2, 2);
    mat << a, 0, 0, a;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::square_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_hexagonal_lattice(double a) {
    using std::numbers::pi;
    using std::numbers::sqrt3;
    bravais_lattice::matrix_type mat(2, 2);
    mat << a / 2, a / 2, -sqrt3 / 2 * a, sqrt3 / 2 * a;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.gamma = 4 * pi / 3;
    p.tag = bravais_lattice::lattice_tag::hexagonal_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_rectangular_lattice(double a, double b) {
    using std::numbers::pi;
    bravais_lattice::matrix_type mat(2, 2);
    mat << a, 0, 0, b;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::rectangular_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_rectangular_centered_lattice(double a, double b) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(2), a2(2);
    a1 << a, 0;
    a2 << 0, b;
    bravais_lattice::matrix_type mat(2, 2);
    mat.col(0) = a1;
    mat.col(1) = (a1 + a2) * 0.5;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::rectangular_centered_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_oblique_lattice(double a, double b, double gamma) {
    bravais_lattice::vector_type a1(2), a2(2);
    a1 << a, 0;
    a2 << std::cos(gamma) * b, std::sin(gamma) * b;
    bravais_lattice::matrix_type mat(2, 2);
    mat.col(0) = a1;
    mat.col(1) = a2;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.gamma = gamma;
    p.tag = bravais_lattice::lattice_tag::oblique_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

void check_3d_lattice_parameters(const bravais_lattice::params& p) {
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0", "check_3d_lattice_parameters");
    }
    if (p.b <= 0.0) {
        throw simplemc_exception("b <= 0", "check_3d_lattice_parameters");
    }
    if (p.c <= 0.0) {
        throw simplemc_exception("c <= 0", "check_3d_lattice_parameters");
    }
    if (simplemc::abs_diff(std::sin(p.alpha), 0.0) < 1e-14) {
        throw simplemc_exception("alpha is a multiple of pi", "check_3d_lattice_parameters");
    }
    if (simplemc::abs_diff(std::sin(p.beta), 0.0) < 1e-14) {
        throw simplemc_exception("beta is a multiple of pi", "check_3d_lattice_parameters");
    }
    if (simplemc::abs_diff(std::sin(p.gamma), 0.0) < 1e-14) {
        throw simplemc_exception("gamma is a multiple of pi", "check_3d_lattice_parameters");
    }
}

bravais_lattice make_cubic_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice::matrix_type mat(3, 3);
    mat << a, 0, 0, 0, a, 0, 0, 0, a;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = a;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::cubic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_fcc_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << 0, a / 2, a / 2;
    a2 << a / 2, 0, a / 2;
    a3 << a / 2, a / 2, 0;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = a;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::fcc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_bcc_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << -a / 2, a / 2, a / 2;
    a2 << a / 2, -a / 2, a / 2;
    a3 << a / 2, a / 2, -a / 2;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = a;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::bcc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_hexagonal_lattice(double a, double c) {
    using std::numbers::pi;
    using std::numbers::sqrt3;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a / 2, -sqrt3 * a / 2, 0;
    a2 << a / 2, sqrt3 * a / 2, 0;
    a3 << 0, 0, c;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = 4 * pi / 3;
    p.tag = bravais_lattice::lattice_tag::hexagonal_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_rhombohedral_lattice(double a, double c) {
    using std::numbers::pi;
    using std::numbers::sqrt3;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a / 2, -a / 2 / sqrt3, c / 3;
    a2 << 0, a / sqrt3, c / 3;
    a3 << -a / 2, -a / 2 / sqrt3, c / 3;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = 4 * pi / 3;
    p.tag = bravais_lattice::lattice_tag::rhombohedral_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_tetragonal_lattice(double a, double c) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a, 0, 0;
    a2 << 0, a, 0;
    a3 << 0, 0, c;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::tetragonal_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_tetragonal_bc_lattice(double a, double c) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << -a / 2, a / 2, c / 2;
    a2 << a / 2, -a / 2, c / 2;
    a3 << a / 2, a / 2, -c / 2;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::tetragonal_bc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_orthorhombic_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a, 0, 0;
    a2 << 0, b, 0;
    a3 << 0, 0, c;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::orthorhombic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_orthorhombic_fc_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << 0, b / 2, c / 2;
    a2 << a / 2, 0, c / 2;
    a3 << a / 2, b / 2, 0;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::orthorhombic_fc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_orthorhombic_bc_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << -a / 2, b / 2, c / 2;
    a2 << a / 2, -b / 2, c / 2;
    a3 << a / 2, b / 2, -c / 2;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::orthorhombic_bc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_orthorhombic_base_centered_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a / 2, -b / 2, 0;
    a2 << a / 2, b / 2, 0;
    a3 << 0, 0, c;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::orthorhombic_base_centered_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_monoclinic_lattice(double a, double b, double c, double beta) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a, 0, 0;
    a2 << 0, b, 0;
    a3 << c * std::cos(beta), 0, c * std::sin(beta);
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = beta;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::monoclinic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_monoclinic_base_centered_lattice(double a, double b, double c, double beta) {
    using std::numbers::pi;
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    a1 << a / 2, -b / 2, 0;
    a2 << a / 2, b / 2, 0;
    a3 << c * std::cos(beta), 0, c * std::sin(beta);
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = beta;
    p.gamma = pi / 2;
    p.tag = bravais_lattice::lattice_tag::monoclinic_base_centered_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice make_triclinic_lattice(double a, double b, double c, double alpha, double beta, double gamma) {
    bravais_lattice::vector_type a1(3), a2(3), a3(3);
    double cx = c * std::cos(beta);
    double cy = c * (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) / std::sin(gamma);
    double cz = std::sqrt(c * c + cx * cx + cy * cy);
    a1 << a, 0, 0;
    a2 << b * std::cos(gamma), b * std::sin(gamma), 0;
    a3 << cx, cy, cz;
    bravais_lattice::matrix_type mat(3, 3);
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice::params p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = alpha;
    p.beta = beta;
    p.gamma = gamma;
    p.tag = bravais_lattice::lattice_tag::triclinic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

} // namespace simplemc
