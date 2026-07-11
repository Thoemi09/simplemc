// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice_1d.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_1d.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <utility>

namespace simplemc {

namespace detail {

void check_1d_lattice_parameters(const lattice_parameters& p) {
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0");
    }
}

} // namespace detail

std::pair<bravais_lattice<1>, lattice_parameters> make_linear_lattice(double a) {
    using enum simplemc::lattice_tag;

    // prepare lattice parameters
    lattice_parameters p { .a = a, .tag = linear_1d };
    detail::check_1d_lattice_parameters(p);

    // prepare basis vectors
    bravais_lattice<1>::matrix_type A;
    A << a;

    return { A, p };
}

} // namespace simplemc
