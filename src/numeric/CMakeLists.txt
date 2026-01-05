# Build numeric library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-numeric (simplemc::simplemc-numeric): Bravais lattices, interpolation, quadrature, etc.

add_library(simplemc-numeric)
add_library(simplemc::simplemc-numeric ALIAS simplemc-numeric)
target_sources(simplemc-numeric
    PRIVATE
        bravais_lattice.cpp
        bravais_lattice_1d.cpp
        bravais_lattice_2d.cpp
        bravais_lattice_3d.cpp
        laguerre_polynomial.cpp
        hermite_polynomial.cpp)
target_link_libraries(simplemc-numeric
    PUBLIC
        simplemc::simplemc-grids
        Eigen3::Eigen)
