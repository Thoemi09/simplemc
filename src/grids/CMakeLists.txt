# Build grids library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-grids (simplemc::simplemc-grids): Various grids in R^1 and R^N.

add_library(simplemc-grids INTERFACE)
add_library(simplemc::simplemc-grids ALIAS simplemc-grids)
target_link_libraries(simplemc-grids
    INTERFACE
        simplemc::simplemc-utils
        simplemc::simplemc-serialize-core)
