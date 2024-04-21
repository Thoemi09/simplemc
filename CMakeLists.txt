# User options/variables:
# ------------------------
# SIMPLEMC_ENABLE_INSTALL: Enable install target (only if no dependencies are fetched).
# SIMPLEMC_BUILD_TESTS: Build unit tests with googletest.
# SIMPLEMC_BUILD_EXAMPLES: Build examples.
# SIMPLEMC_BUILD_DOC: Build documentation with Doxygen.
# SIMPLEMC_WITH_HIGHFIVE: Use HighFive for HDF5 support.
# SIMPLEMC_FETCH_FMT: Fetch fmt from github.
# SIMPLEMC_FETCH_RANGEV3: Fetch range-v3 from github.
# SIMPLEMC_FETCH_EIGEN3: Fetch Eigen3 from gitlab.
# SIMPLEMC_FETCH_NLOHMANN_JSON: Fetch nlohmann_json library from github.
# SIMPLEMC_FETCH_HIGHFIVE: Fetch HighFive library from github.
# SIMPLEMC_DISABLE_WARNINGS: Disable stricter compilation warnings.

cmake_minimum_required(VERSION 3.24)
project(simplemc
    VERSION 0.1.0
    LANGUAGES CXX)

#########################
# Project wide setup
#########################
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(BuildAndInstallDirs)

if(NOT DEFINED SIMPLEMC_MASTER_PROJECT)
    set(SIMPLEMC_MASTER_PROJECT OFF)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(SIMPLEMC_MASTER_PROJECT ON)
    endif()
endif()

#########################
# User/Build options
#########################
option(SIMPLEMC_ENABLE_INSTALL "Enable install target" ${SIMPLEMC_MASTER_PROJECT})
option(SIMPLEMC_BUILD_TESTS "Build unit tests" ${SIMPLEMC_MASTER_PROJECT})
option(SIMPLEMC_BUILD_EXAMPLES "Build examples" ${SIMPLEMC_MASTER_PROJECT})
option(SIMPLEMC_BUILD_DOC "Build documentation with Doxygen" OFF)
option(SIMPLEMC_WITH_HIGHFIVE "Use HighFive for HDF5 support" ON)
option(SIMPLEMC_FETCH_FMT "Fetch fmt from github" ON)
option(SIMPLEMC_FETCH_RANGEV3 "Fetch range-v3 from github" ON)
option(SIMPLEMC_FETCH_EIGEN3 "Fetch Eigen3 from gitlab" ON)
option(SIMPLEMC_FETCH_NLOHMANN_JSON "Fetch nlohmann_json library from github" ON)
option(SIMPLEMC_FETCH_HIGHFIVE "Fetch HighFive library from github" ON)
option(SIMPLEMC_DISABLE_WARNINGS "Disable stricter compilation warnings" OFF)

# output user options
message(STATUS "Start configuring project ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Enable install: ${SIMPLEMC_ENABLE_INSTALL}")
message(STATUS "Build unit tests: ${SIMPLEMC_BUILD_TESTS}")
message(STATUS "Build examples: ${SIMPLEMC_BUILD_EXAMPLES}")
message(STATUS "Build documentation: ${SIMPLEMC_BUILD_DOC}")
message(STATUS "HDF5 support: ${SIMPLEMC_WITH_HIGHFIVE}")
message(STATUS "Fetch fmt: ${SIMPLEMC_FETCH_FMT}")
message(STATUS "Fetch range-v3: ${SIMPLEMC_FETCH_RANGEV3}")
message(STATUS "Fetch Eigen3: ${SIMPLEMC_FETCH_EIGEN3}")
message(STATUS "Fetch nlohmann_json: ${SIMPLEMC_FETCH_NLOHMANN_JSON}")
message(STATUS "Fetch HighFive: ${SIMPLEMC_FETCH_HIGHFIVE}")
message(STATUS "Disable stricter compiler warnings: ${SIMPLEMC_DISABLE_WARNINGS}")

#########################
# Build targets
#########################
add_subdirectory(deps)
add_subdirectory(src)
if(SIMPLEMC_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(SIMPLEMC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(SIMPLEMC_BUILD_DOC AND SIMPLEMC_MASTER_PROJECT)
    add_subdirectory(doc)
endif()
