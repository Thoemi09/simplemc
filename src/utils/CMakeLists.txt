# Build utilities library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-utils (simplemc::simplemc-utils): Various utilities used throughout simplemc.

add_library(simplemc-utils)
add_library(simplemc::simplemc-utils ALIAS simplemc-utils)
target_sources(simplemc-utils
    PRIVATE
        generic_error.cpp
        file_io.cpp)
target_link_libraries(simplemc-utils
    PUBLIC
        simplemc::simplemc-basic
        fmt::fmt
        range-v3::range-v3)
