# Build libraries for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-basic (simplemc::simplemc-basic)

# generate config.hpp header file
configure_file(config.hpp.in ${PROJECT_BINARY_DIR}/include/simplemc/config.hpp)

# add basic interface library
add_library(simplemc-basic INTERFACE)
add_library(simplemc::simplemc-basic ALIAS simplemc-basic)
target_compile_options(simplemc-basic 
    INTERFACE
        -Wall
        -Wextra
        -Wfloat-conversion
        -Wpedantic)
target_include_directories(simplemc-basic
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# build simplemc libraries
add_subdirectory(utils)
add_subdirectory(json)
add_subdirectory(random)
add_subdirectory(grids)
add_subdirectory(mpi)
add_subdirectory(numeric)