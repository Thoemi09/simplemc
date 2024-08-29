/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice_3d.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_3d.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <numbers>

namespace simplemc {

void check_3d_lattice_parameters(const lattice_parameters& p) {
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

bravais_lattice<3> make_cubic_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice<3>::matrix_type mat;
    mat << a, 0, 0, 0, a, 0, 0, 0, a;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = a;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::cubic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_fcc_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << 0, a / 2, a / 2;
    a2 << a / 2, 0, a / 2;
    a3 << a / 2, a / 2, 0;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = a;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::fcc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_bcc_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << -a / 2, a / 2, a / 2;
    a2 << a / 2, -a / 2, a / 2;
    a3 << a / 2, a / 2, -a / 2;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = a;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::bcc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_hexagonal_lattice(double a, double c) {
    using std::numbers::pi;
    using std::numbers::sqrt3;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -sqrt3 * a / 2, 0;
    a2 << a / 2, sqrt3 * a / 2, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = 4 * pi / 3;
    p.tag = lattice_tag::hexagonal_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_rhombohedral_lattice(double a, double c) {
    using std::numbers::pi;
    using std::numbers::sqrt3;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -a / 2 / sqrt3, c / 3;
    a2 << 0, a / sqrt3, c / 3;
    a3 << -a / 2, -a / 2 / sqrt3, c / 3;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = 4 * pi / 3;
    p.tag = lattice_tag::rhombohedral_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_tetragonal_lattice(double a, double c) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a, 0, 0;
    a2 << 0, a, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::tetragonal_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_tetragonal_bc_lattice(double a, double c) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << -a / 2, a / 2, c / 2;
    a2 << a / 2, -a / 2, c / 2;
    a3 << a / 2, a / 2, -c / 2;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = a;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::tetragonal_bc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_orthorhombic_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a, 0, 0;
    a2 << 0, b, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::orthorhombic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_orthorhombic_fc_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << 0, b / 2, c / 2;
    a2 << a / 2, 0, c / 2;
    a3 << a / 2, b / 2, 0;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::orthorhombic_fc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_orthorhombic_bc_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << -a / 2, b / 2, c / 2;
    a2 << a / 2, -b / 2, c / 2;
    a3 << a / 2, b / 2, -c / 2;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::orthorhombic_bc_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_orthorhombic_base_centered_lattice(double a, double b, double c) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -b / 2, 0;
    a2 << a / 2, b / 2, 0;
    a3 << 0, 0, c;
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = pi / 2;
    p.gamma = pi / 2;
    p.tag = lattice_tag::orthorhombic_base_centered_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_monoclinic_lattice(double a, double b, double c, double beta) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a, 0, 0;
    a2 << 0, b, 0;
    a3 << c * std::cos(beta), 0, c * std::sin(beta);
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = beta;
    p.gamma = pi / 2;
    p.tag = lattice_tag::monoclinic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_monoclinic_base_centered_lattice(double a, double b, double c, double beta) {
    using std::numbers::pi;
    bravais_lattice<3>::vector_type a1, a2, a3;
    a1 << a / 2, -b / 2, 0;
    a2 << a / 2, b / 2, 0;
    a3 << c * std::cos(beta), 0, c * std::sin(beta);
    bravais_lattice<3>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    mat.col(2) = a3;
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = pi / 2;
    p.beta = beta;
    p.gamma = pi / 2;
    p.tag = lattice_tag::monoclinic_base_centered_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<3> make_triclinic_lattice(double a, double b, double c, double alpha, double beta, double gamma) {
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
    bravais_lattice<3>::param_type p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.alpha = alpha;
    p.beta = beta;
    p.gamma = gamma;
    p.tag = lattice_tag::triclinic_3d;
    check_3d_lattice_parameters(p);
    return { mat, p };
}

} // namespace simplemc
