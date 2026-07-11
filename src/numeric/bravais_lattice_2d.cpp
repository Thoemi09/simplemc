// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice_2d.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_2d.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <numbers>
#include <utility>

namespace simplemc {

namespace detail {

void check_2d_lattice_parameters(const lattice_parameters& p) {
    using std::numbers::pi;
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0");
    }
    if (p.b <= 0.0) {
        throw simplemc_exception("b <= 0");
    }
    if (p.gamma <= 0 || p.gamma > pi) {
        throw simplemc_exception("gamma is not in (0, pi]");
    }
}

} // namespace detail

std::pair<bravais_lattice<2>, lattice_parameters> make_square_lattice(double a) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;

    // prepare lattice parameters
    lattice_parameters p { .a = a, .b = a, .gamma = pi / 2, .tag = square_2d };
    detail::check_2d_lattice_parameters(p);

    // prepare basis vectors
    bravais_lattice<2>::matrix_type A;
    A << a, 0, 0, a;

    return { A, p };
}

std::pair<bravais_lattice<2>, lattice_parameters> make_hexagonal_lattice(double a) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;
    using std::numbers::sqrt3;

    // prepare lattice parameters
    lattice_parameters p { .a = a, .b = a, .gamma = 2 * pi / 3, .tag = hexagonal_2d };
    detail::check_2d_lattice_parameters(p);

    // prepare basis vectors
    bravais_lattice<2>::matrix_type A;
    A << a / 2, a / 2, -sqrt3 / 2 * a, sqrt3 / 2 * a;

    return { A, p };
}

std::pair<bravais_lattice<2>, lattice_parameters> make_rectangular_lattice(double a, double b) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;

    // prepare lattice parameters
    lattice_parameters p { .a = a, .b = b, .gamma = pi / 2, .tag = rectangular_2d };
    detail::check_2d_lattice_parameters(p);

    // prepare basis vectors
    bravais_lattice<2>::matrix_type A;
    A << a, 0, 0, b;

    return { A, p };
}

std::pair<bravais_lattice<2>, lattice_parameters> make_rectangular_centered_lattice(double a, double b) {
    using enum simplemc::lattice_tag;
    using std::numbers::pi;

    // prepare lattice parameters
    lattice_parameters p { .a = a, .b = b, .gamma = pi / 2, .tag = rectangular_centered_2d };
    detail::check_2d_lattice_parameters(p);

    // prepare basis vectors
    bravais_lattice<2>::vector_type a1, a2;
    a1 << a, 0;
    a2 << 0, b;
    bravais_lattice<2>::matrix_type A;
    A.col(0) = a1;
    A.col(1) = (a1 + a2) * 0.5;

    return { A, p };
}

std::pair<bravais_lattice<2>, lattice_parameters> make_oblique_lattice(double a, double b, double gamma) {
    using enum simplemc::lattice_tag;

    // prepare lattice parameters
    lattice_parameters p { .a = a, .b = b, .gamma = gamma, .tag = oblique_2d };
    detail::check_2d_lattice_parameters(p);

    // prepare basis vectors
    bravais_lattice<2>::vector_type a1, a2;
    a1 << a, 0;
    a2 << std::cos(gamma) * b, std::sin(gamma) * b;
    bravais_lattice<2>::matrix_type A;
    A.col(0) = a1;
    A.col(1) = a2;

    return { A, p };
}

} // namespace simplemc
