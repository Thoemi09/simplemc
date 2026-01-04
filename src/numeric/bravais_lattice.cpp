/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <string>
#include <string_view>

namespace simplemc {

std::string lattice_tag_to_string(const lattice_tag& tag) {
    using enum lattice_tag;
    switch (tag) {
        case linear_1d: return "linear_1d";
        case square_2d: return "square_2d";
        case hexagonal_2d: return "hexagonal_2d";
        case rectangular_2d: return "rectangular_2d";
        case rectangular_centered_2d: return "rectangular_centered_2d";
        case oblique_2d: return "oblique_2d";
        case cubic_3d: return "cubic_3d";
        case fcc_3d: return "fcc_3d";
        case bcc_3d: return "bcc_3d";
        case hexagonal_3d: return "hexagonal_3d";
        case rhombohedral_3d: return "rhombohedral_3d";
        case tetragonal_3d: return "tetragonal_3d";
        case tetragonal_bc_3d: return "tetragonal_bc_3d";
        case orthorhombic_3d: return "orthorhombic_3d";
        case orthorhombic_fc_3d: return "orthorhombic_fc_3d";
        case orthorhombic_bc_3d: return "orthorhombic_bc_3d";
        case orthorhombic_base_centered_3d: return "orthorhombic_base_centered_3d";
        case monoclinic_3d: return "monoclinic_3d";
        case monoclinic_base_centered_3d: return "monoclinic_base_centered_3d";
        case triclinic_3d: return "triclinic_3d";
        case unspecified: return "unspecified";
        default: throw simplemc_exception("Unknown lattice tag");
    }
}

lattice_tag string_to_lattice_tag(std::string_view str) {
    if (str == "linear_1d") {
        return lattice_tag::linear_1d;
    }
    if (str == "square_2d") {
        return lattice_tag::square_2d;
    }
    if (str == "hexagonal_2d") {
        return lattice_tag::hexagonal_2d;
    }
    if (str == "rectangular_2d") {
        return lattice_tag::rectangular_2d;
    }
    if (str == "rectangular_centered_2d") {
        return lattice_tag::rectangular_centered_2d;
    }
    if (str == "oblique_2d") {
        return lattice_tag::oblique_2d;
    }
    if (str == "cubic_3d") {
        return lattice_tag::cubic_3d;
    }
    if (str == "fcc_3d") {
        return lattice_tag::fcc_3d;
    }
    if (str == "bcc_3d") {
        return lattice_tag::bcc_3d;
    }
    if (str == "hexagonal_3d") {
        return lattice_tag::hexagonal_3d;
    }
    if (str == "rhombohedral_3d") {
        return lattice_tag::rhombohedral_3d;
    }
    if (str == "tetragonal_3d") {
        return lattice_tag::tetragonal_3d;
    }
    if (str == "tetragonal_bc_3d") {
        return lattice_tag::tetragonal_bc_3d;
    }
    if (str == "orthorhombic_3d") {
        return lattice_tag::orthorhombic_3d;
    }
    if (str == "orthorhombic_fc_3d") {
        return lattice_tag::orthorhombic_fc_3d;
    }
    if (str == "orthorhombic_bc_3d") {
        return lattice_tag::orthorhombic_bc_3d;
    }
    if (str == "orthorhombic_base_centered_3d") {
        return lattice_tag::orthorhombic_base_centered_3d;
    }
    if (str == "monoclinic_3d") {
        return lattice_tag::monoclinic_3d;
    }
    if (str == "monoclinic_base_centered_3d") {
        return lattice_tag::monoclinic_base_centered_3d;
    }
    if (str == "triclinic_3d") {
        return lattice_tag::triclinic_3d;
    }
    if (str == "unspecified") {
        return lattice_tag::unspecified;
    }
    throw simplemc_exception(fmt::format("Unknown lattice string: {}", str));
}

} // namespace simplemc
