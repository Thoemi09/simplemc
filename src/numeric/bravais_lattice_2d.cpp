/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice_2d.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_2d.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <numbers>

namespace simplemc {

void check_2d_lattice_parameters(const lattice_parameters& p) {
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

bravais_lattice<2> make_square_lattice(double a) {
    using std::numbers::pi;
    bravais_lattice<2>::matrix_type mat;
    mat << a, 0, 0, a;
    bravais_lattice<2>::param_type p;
    p.a = a;
    p.b = a;
    p.gamma = pi / 2;
    p.tag = lattice_tag::square_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<2> make_hexagonal_lattice(double a) {
    using std::numbers::pi;
    using std::numbers::sqrt3;
    bravais_lattice<2>::matrix_type mat;
    mat << a / 2, a / 2, -sqrt3 / 2 * a, sqrt3 / 2 * a;
    bravais_lattice<2>::param_type p;
    p.a = a;
    p.b = a;
    p.gamma = 4 * pi / 3;
    p.tag = lattice_tag::hexagonal_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<2> make_rectangular_lattice(double a, double b) {
    using std::numbers::pi;
    bravais_lattice<2>::matrix_type mat;
    mat << a, 0, 0, b;
    bravais_lattice<2>::param_type p;
    p.a = a;
    p.b = b;
    p.gamma = pi / 2;
    p.tag = lattice_tag::rectangular_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<2> make_rectangular_centered_lattice(double a, double b) {
    using std::numbers::pi;
    bravais_lattice<2>::vector_type a1, a2;
    a1 << a, 0;
    a2 << 0, b;
    bravais_lattice<2>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = (a1 + a2) * 0.5;
    bravais_lattice<2>::param_type p;
    p.a = a;
    p.b = b;
    p.gamma = pi / 2;
    p.tag = lattice_tag::rectangular_centered_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

bravais_lattice<2> make_oblique_lattice(double a, double b, double gamma) {
    bravais_lattice<2>::vector_type a1, a2;
    a1 << a, 0;
    a2 << std::cos(gamma) * b, std::sin(gamma) * b;
    bravais_lattice<2>::matrix_type mat;
    mat.col(0) = a1;
    mat.col(1) = a2;
    bravais_lattice<2>::param_type p;
    p.a = a;
    p.b = b;
    p.gamma = gamma;
    p.tag = lattice_tag::oblique_2d;
    check_2d_lattice_parameters(p);
    return { mat, p };
}

} // namespace simplemc
