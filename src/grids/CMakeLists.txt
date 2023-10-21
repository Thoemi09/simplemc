# Build grids library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-grids (simplemc::simplemc-grids)

add_library(simplemc-grids)
add_library(simplemc::simplemc-grids ALIAS simplemc-grids)
target_sources(simplemc-grids 
    PRIVATE 
        grid_base.cpp
        linear_grid.cpp
        power_grid.cpp
        symmetric_power_grid.cpp)
target_link_libraries(simplemc-grids 
    PUBLIC 
        simplemc::simplemc-utils)