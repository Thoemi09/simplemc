# Build utilities library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# utils (simplemc::utils)

add_library(utils)
add_library(simplemc::utils ALIAS utils)
target_sources(utils 
    PRIVATE 
        generic_error.cpp
        file_io.cpp)
target_link_libraries(utils 
    PUBLIC 
        simplemc::basic 
        fmt::fmt)