# Build numeric library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-numeric (simplemc::simplemc-numeric): Bravais lattices, interpolation, quadrature, etc.

add_library(simplemc-numeric)
add_library(simplemc::simplemc-numeric ALIAS simplemc-numeric)
target_sources(simplemc-numeric
    PRIVATE
        utils.cpp
        bravais_lattice.cpp
        linear_map.cpp
        orthogonal_polynomials.cpp)
target_link_libraries(simplemc-numeric
    PUBLIC
        simplemc::simplemc-grids
        Eigen3::Eigen)
