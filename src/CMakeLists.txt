# Build libraries for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-basic (simplemc::simplemc-basic): Interface library specifying compiler warnings and
# include directories.
# simplemc (simplemc::simplemc): Interface library containing all simplemc sublibraries.

# generate config.hpp header file
configure_file(config.hpp.in ${PROJECT_BINARY_DIR}/include/simplemc/config.hpp)

# add basic interface library
add_library(simplemc-basic INTERFACE)
add_library(simplemc::simplemc-basic ALIAS simplemc-basic)
if(NOT SIMPLEMC_DISABLE_WARNINGS)
    target_compile_options(simplemc-basic
        INTERFACE
            -Wall
            -Wextra
            -Wfloat-conversion
            -Wpedantic
            -Wno-psabi)
else()
    target_compile_options(simplemc-basic
        INTERFACE
            -Wno-psabi)
endif()
target_include_directories(simplemc-basic
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# build simplemc sublibraries
add_subdirectory(utils)
add_subdirectory(mpi)
add_subdirectory(random)
add_subdirectory(grids)
add_subdirectory(numeric)
add_subdirectory(accs)
add_subdirectory(serialize)

# add simplemc library
add_library(simplemc INTERFACE)
add_library(simplemc::simplemc ALIAS simplemc)
target_link_libraries(simplemc
    INTERFACE
        simplemc::simplemc-utils
        simplemc::simplemc-mpi
        simplemc::simplemc-random
        simplemc::simplemc-grids
        simplemc::simplemc-numeric
        simplemc::simplemc-accs
        simplemc::simplemc-serialize)
