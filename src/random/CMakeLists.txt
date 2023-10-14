# Build random library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-random (simplemc::simplemc-random)

add_library(simplemc-random)
add_library(simplemc::simplemc-random ALIAS simplemc-random)
target_sources(simplemc-random 
    PRIVATE 
        splitmix64.cpp
        xoshiro256.cpp
        uniform_real_distribution.cpp)
target_link_libraries(simplemc-random 
    PUBLIC 
        simplemc::simplemc-utils)