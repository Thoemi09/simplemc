/**
 * @file
 * @brief Implementation details for simplemc/numeric/bravais_lattice.hpp.
 */

#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <string>

namespace simplemc {

std::string lattice_tag_to_string(const lattice_tag& tag) {
    if (tag == lattice_tag::linear_1d) {
        return "linear_1d";
    } else if (tag == lattice_tag::square_2d) {
        return "square_2d";
    } else if (tag == lattice_tag::hexagonal_2d) {
        return "hexagonal_2d";
    } else if (tag == lattice_tag::rectangular_2d) {
        return "rectangular_2d";
    } else if (tag == lattice_tag::rectangular_centered_2d) {
        return "rectangular_centered_2d";
    } else if (tag == lattice_tag::oblique_2d) {
        return "oblique_2d";
    } else if (tag == lattice_tag::cubic_3d) {
        return "cubic_3d";
    } else if (tag == lattice_tag::fcc_3d) {
        return "fcc_3d";
    } else if (tag == lattice_tag::bcc_3d) {
        return "bcc_3d";
    } else if (tag == lattice_tag::hexagonal_3d) {
        return "hexagonal_3d";
    } else if (tag == lattice_tag::rhombohedral_3d) {
        return "rhombohedral_3d";
    } else if (tag == lattice_tag::tetragonal_3d) {
        return "tetragonal_3d";
    } else if (tag == lattice_tag::tetragonal_bc_3d) {
        return "tetragonal_bc_3d";
    } else if (tag == lattice_tag::orthorhombic_3d) {
        return "orthorhombic_3d";
    } else if (tag == lattice_tag::orthorhombic_fc_3d) {
        return "orthorhombic_fc_3d";
    } else if (tag == lattice_tag::orthorhombic_bc_3d) {
        return "orthorhombic_bc_3d";
    } else if (tag == lattice_tag::orthorhombic_base_centered_3d) {
        return "orthorhombic_base_centered_3d";
    } else if (tag == lattice_tag::monoclinic_3d) {
        return "monoclinic_3d";
    } else if (tag == lattice_tag::monoclinic_base_centered_3d) {
        return "monoclinic_base_centered_3d";
    } else if (tag == lattice_tag::triclinic_3d) {
        return "triclinic_3d";
    } else if (tag == lattice_tag::unspecified) {
        return "unspecified";
    } else {
        throw simplemc_exception("Wrong lattice tag", "lattice_tag_to_string");
    }
}

lattice_tag string_to_lattice_tag(const std::string& str) {
    if (str == "linear_1d") {
        return lattice_tag::linear_1d;
    } else if (str == "square_2d") {
        return lattice_tag::square_2d;
    } else if (str == "hexagonal_2d") {
        return lattice_tag::hexagonal_2d;
    } else if (str == "rectangular_2d") {
        return lattice_tag::rectangular_2d;
    } else if (str == "rectangular_centered_2d") {
        return lattice_tag::rectangular_centered_2d;
    } else if (str == "oblique_2d") {
        return lattice_tag::oblique_2d;
    } else if (str == "cubic_3d") {
        return lattice_tag::cubic_3d;
    } else if (str == "fcc_3d") {
        return lattice_tag::fcc_3d;
    } else if (str == "bcc_3d") {
        return lattice_tag::bcc_3d;
    } else if (str == "hexagonal_3d") {
        return lattice_tag::hexagonal_3d;
    } else if (str == "rhombohedral_3d") {
        return lattice_tag::rhombohedral_3d;
    } else if (str == "tetragonal_3d") {
        return lattice_tag::tetragonal_3d;
    } else if (str == "tetragonal_bc_3d") {
        return lattice_tag::tetragonal_bc_3d;
    } else if (str == "orthorhombic_3d") {
        return lattice_tag::orthorhombic_3d;
    } else if (str == "orthorhombic_fc_3d") {
        return lattice_tag::orthorhombic_fc_3d;
    } else if (str == "orthorhombic_bc_3d") {
        return lattice_tag::orthorhombic_bc_3d;
    } else if (str == "orthorhombic_base_centered_3d") {
        return lattice_tag::orthorhombic_base_centered_3d;
    } else if (str == "monoclinic_3d") {
        return lattice_tag::monoclinic_3d;
    } else if (str == "monoclinic_base_centered_3d") {
        return lattice_tag::monoclinic_base_centered_3d;
    } else if (str == "triclinic_3d") {
        return lattice_tag::triclinic_3d;
    } else if (str == "unspecified") {
        return lattice_tag::unspecified;
    } else {
        throw simplemc_exception(fmt::format("Wrong lattice string: {}", str), "string_to_lattice_tag");
    }
}

} // namespace simplemc
