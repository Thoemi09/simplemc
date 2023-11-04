# Build numeric library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-numeric (simplemc::simplemc-numeric)

add_library(simplemc-numeric)
add_library(simplemc::simplemc-numeric ALIAS simplemc-numeric)
target_sources(simplemc-numeric
    PRIVATE
        utils.cpp
        bravais_lattice.cpp)
target_link_libraries(simplemc-numeric 
    PUBLIC 
        simplemc::simplemc-grids 
        Eigen3::Eigen)
