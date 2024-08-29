/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice_1d.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/bravais_lattice_1d.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

void check_1d_lattice_parameters(const lattice_parameters& p) {
    if (p.a <= 0.0) {
        throw simplemc_exception("a <= 0", "check_1d_lattice_parameters");
    }
}

bravais_lattice<1> make_linear_lattice(double a) {
    bravais_lattice<1>::matrix_type mat;
    mat << a;
    bravais_lattice<1>::param_type p;
    p.a = a;
    p.tag = lattice_tag::linear_1d;
    check_1d_lattice_parameters(p);
    return { mat, p };
}

} // namespace simplemc
