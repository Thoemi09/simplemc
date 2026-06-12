# Build mc library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-mc (simplemc::simplemc-mc): Monte Carlo simulation building blocks.

add_library(simplemc-mc INTERFACE)
add_library(simplemc::simplemc-mc ALIAS simplemc-mc)
target_link_libraries(simplemc-mc
    INTERFACE
        simplemc::simplemc-utils
        simplemc::simplemc-mpi
        simplemc::simplemc-serialize)
