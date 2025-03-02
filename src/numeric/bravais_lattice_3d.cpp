/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice_3d.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_3d.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <numbers>
#include <utility>

namespace simplemc {

namespace detail {

void check_3d_lattice_parameters(const lattice_parameters& p) {
    using std::numbers::pi;
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0", "detail::check_3d_lattice_parameters");
    }
    if (p.b <= 0.0) {
        throw simplemc_exception("b <= 0", "detail::check_3d_lattice_parameters");
    }
    if (p.c <= 0.0) {
        throw simplemc_exception("c <= 0", "detail::check_3d_lattice_parameters");
    }
    if (p.alpha <= 0 || p.alpha > pi) {
        throw simplemc_exception("alpha is not in (0, pi]", "detail::check_3d_lattice_parameters");
    }
    if (p.beta <= 0 || p.beta > pi) {
        throw simplemc_exception("beta is not in (0, pi]", "detail::check_3d_lattice_parameters");
    }
    if (p.gamma <= 0 || p.gamma > pi) {
        throw simplemc_exception("gamma is not in (0, pi]", "detail::check_3d_lattice_parameters");
    }
}

} // namespace detail

std::pair<bravais_lattice<3>, lattice_parameters> make_cubic_lattice(double a) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p { .a = a, .b = a, .c = a, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = cubic_3d };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::matrix_type mat;
    mat << a, 0, 0, 0, a, 0, 0, 0, a;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_fcc_lattice(double a) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p { .a = a, .b = a, .c = a, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = fcc_3d };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << 0, a / 2, a / 2;
    a2 << a / 2, 0, a / 2;
    a3 << a / 2, a / 2, 0;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_bcc_lattice(double a) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p { .a = a, .b = a, .c = a, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = bcc_3d };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << -a / 2, a / 2, a / 2;
    a2 << a / 2, -a / 2, a / 2;
    a3 << a / 2, a / 2, -a / 2;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_hexagonal_lattice(double a, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    using std::numbers::sqrt3;
    lattice_parameters p {
        .a = a, .b = a, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = 2 * pi / 3, .tag = hexagonal_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -sqrt3 * a / 2, 0;
    a2 << a / 2, sqrt3 * a / 2, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_rhombohedral_lattice(double a, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    using std::numbers::sqrt3;
    lattice_parameters p {
        .a = a, .b = a, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = 2 * pi / 3, .tag = rhombohedral_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -a / 2 / sqrt3, c / 3;
    a2 << 0, a / sqrt3, c / 3;
    a3 << -a / 2, -a / 2 / sqrt3, c / 3;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_tetragonal_lattice(double a, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = a, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = tetragonal_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a, 0, 0;
    a2 << 0, a, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_tetragonal_bc_lattice(double a, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = a, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = tetragonal_bc_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << -a / 2, a / 2, c / 2;
    a2 << a / 2, -a / 2, c / 2;
    a3 << a / 2, a / 2, -c / 2;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_lattice(double a, double b, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = b, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = orthorhombic_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a, 0, 0;
    a2 << 0, b, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_fc_lattice(double a, double b, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = b, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = orthorhombic_fc_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << 0, b / 2, c / 2;
    a2 << a / 2, 0, c / 2;
    a3 << a / 2, b / 2, 0;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_bc_lattice(double a, double b, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = b, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = orthorhombic_bc_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << -a / 2, b / 2, c / 2;
    a2 << a / 2, -b / 2, c / 2;
    a3 << a / 2, b / 2, -c / 2;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_orthorhombic_base_centered_lattice(
    double a, double b, double c) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = b, .c = c, .alpha = pi / 2, .beta = pi / 2, .gamma = pi / 2, .tag = orthorhombic_base_centered_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -b / 2, 0;
    a2 << a / 2, b / 2, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_monoclinic_lattice(double a, double b, double c, double beta) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = b, .c = c, .alpha = pi / 2, .beta = beta, .gamma = pi / 2, .tag = monoclinic_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a, 0, 0;
    a2 << 0, b, 0;
    a3 << c * std::cos(beta), 0, c * std::sin(beta);
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_monoclinic_base_centered_lattice(
    double a, double b, double c, double beta) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    lattice_parameters p {
        .a = a, .b = b, .c = c, .alpha = pi / 2, .beta = beta, .gamma = pi / 2, .tag = monoclinic_base_centered_3d
    };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -b / 2, 0;
    a2 << a / 2, b / 2, 0;
    a3 << c * std::cos(beta), 0, c * std::sin(beta);
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

std::pair<bravais_lattice<3>, lattice_parameters> make_triclinic_lattice(
    double a, double b, double c, double alpha, double beta, double gamma) {
    using enum simplemc::lattice_tag;
    lattice_parameters p { .a = a, .b = b, .c = c, .alpha = alpha, .beta = beta, .gamma = gamma, .tag = triclinic_3d };
    detail::check_3d_lattice_parameters(p);
    bravais_lattice<3>::vector_type a1, a2, a3;
    double cx = c * std::cos(beta);
    double cy = c * (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) / std::sin(gamma);
    double cz = std::sqrt(c * c + cx * cx + cy * cy);
    a1 << a, 0, 0;
    a2 << b * std::cos(gamma), b * std::sin(gamma), 0;
    a3 << cx, cy, cz;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    return { mat, p };
}

} // namespace simplemc
