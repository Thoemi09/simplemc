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
target_compile_features(simplemc-basic INTERFACE cxx_std_23)
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
add_subdirectory(mc)

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
        simplemc::simplemc-serialize
        simplemc::simplemc-mc)

#########################
# Install rules
#########################
if(SIMPLEMC_ENABLE_INSTALL)
    # refuse to configure an install while any public dependency is being fetched
    if(SIMPLEMC_FETCH_FMT OR SIMPLEMC_FETCH_EIGEN3 OR SIMPLEMC_FETCH_NLOHMANN_JSON 
            OR SIMPLEMC_FETCH_RANGEV3 OR SIMPLEMC_FETCH_HIGHFIVE)
        message(FATAL_ERROR
            "SIMPLEMC_ENABLE_INSTALL is ON but one or more public dependencies are being fetched. "
            "Reconfigure with the relevant SIMPLEMC_FETCH_*=OFF, or set SIMPLEMC_ENABLE_INSTALL=OFF.")
    endif()

    include(CMakePackageConfigHelpers)

    # install all interface + compiled targets into a single export set
    install(
        TARGETS
            simplemc simplemc-basic
            simplemc-utils simplemc-mpi simplemc-random simplemc-grids
            simplemc-numeric simplemc-accs simplemc-mc
            simplemc-serialize-core simplemc-serialize
            simplemc-serialize-variant simplemc-serialize-json
        EXPORT simplemc-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    if(SIMPLEMC_USE_HDF5)
        install(TARGETS simplemc-serialize-hdf5 EXPORT simplemc-targets)
    endif()

    # install headers: shipped source tree + generated config.hpp
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/simplemc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${PROJECT_BINARY_DIR}/include/simplemc/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simplemc)

    # install the export set
    install(EXPORT simplemc-targets
        FILE simplemc-targets.cmake
        NAMESPACE simplemc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simplemc)

    # generate and install the package config files
    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/simplemc-config.cmake.in
        ${PROJECT_BINARY_DIR}/simplemc-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simplemc)

    write_basic_package_version_file(
        ${PROJECT_BINARY_DIR}/simplemc-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(
        FILES
            ${PROJECT_BINARY_DIR}/simplemc-config.cmake
            ${PROJECT_BINARY_DIR}/simplemc-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simplemc)
endif()
