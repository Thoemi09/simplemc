#include <fmt/base.h>
#include <simplemc/numeric.hpp>
#include <simplemc/utils.hpp>

int main() {
    // create an FCC lattice with a lattice constant of 1.5
    auto [fcc, lattice_params] = simplemc::make_fcc_lattice(1.5);

    // print the lattice vectors of the primitive cell
    fmt::println("Lattice vectors:\n{}\n", simplemc::to_string(fcc.real_lattice_vectors()));

    // print the volume of the primitive cell
    fmt::println("Volume: {}\n", fcc.real_cell_volume());

    // print the reciprocal lattice vectors
    fmt::println("Reciprocal lattice vectors:\n{}\n", simplemc::to_string(fcc.reciprocal_lattice_vectors()));

    // print the volume of the reciprocal cell
    fmt::println("Reciprocal volume: {}", fcc.reciprocal_cell_volume());
}
