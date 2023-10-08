# Build libraries for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# basic (simplemc::basic)

# generate config.h header file
configure_file(config.h.in ${PROJECT_BINARY_DIR}/include/simplemc/config.h)

# add basic interface library
add_library(basic INTERFACE)
add_library(simplemc::basic ALIAS basic)
target_compile_options(basic 
    INTERFACE
        -Wall
        -Wextra
        -Wfloat-conversion
        -Wpedantic)
target_include_directories(basic
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# build simplemc libraries
add_subdirectory(utils)