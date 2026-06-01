# Build random library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-random (simplemc::simplemc-random): RNGs and other random utilities.

add_library(simplemc-random INTERFACE)
add_library(simplemc::simplemc-random ALIAS simplemc-random)
target_link_libraries(simplemc-random
    INTERFACE
        simplemc::simplemc-utils
        simplemc::simplemc-serialize-core)
