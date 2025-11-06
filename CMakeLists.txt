# User options/variables:
# ------------------------
# SIMPLEMC_ENABLE_INSTALL: Enable install target.
# SIMPLEMC_BUILD_TESTS: Build unit tests with googletest.
# SIMPLEMC_BUILD_EXAMPLES: Build examples.
# SIMPLEMC_BUILD_DOC: Build documentation with Doxygen.
# SIMPLEMC_USE_STD_RANGES: Use ranges from the standard library.
# SIMPLEMC_FETCH_FMT: Fetch fmt from github.
# SIMPLEMC_FETCH_RANGEV3: Fetch range-v3 from github.
# SIMPLEMC_FETCH_EIGEN3: Fetch Eigen3 from gitlab.
# SIMPLEMC_FETCH_NLOHMANN_JSON: Fetch nlohmann_json from github.
# SIMPLEMC_DISABLE_WARNINGS: Disable stricter compilation warnings.

cmake_minimum_required(VERSION 3.24)
project(simplemc
    VERSION 0.1.0
    LANGUAGES CXX)

#########################
# Project wide setup
#########################
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(BuildAndInstallDirs)

set(SIMPLEMC_MASTER_PROJECT ${PROJECT_IS_TOP_LEVEL})

set(SIMPLEMC_USES_GNU OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(SIMPLEMC_USES_GNU ON)
endif()

#########################
# User/Build options
#########################
option(SIMPLEMC_ENABLE_INSTALL "Enable install target" ${SIMPLEMC_MASTER_PROJECT})
option(SIMPLEMC_BUILD_TESTS "Build unit tests with googletest" ${SIMPLEMC_MASTER_PROJECT})
option(SIMPLEMC_BUILD_EXAMPLES "Build examples" ${SIMPLEMC_MASTER_PROJECT})
option(SIMPLEMC_BUILD_DOC "Build documentation with Doxygen" OFF)
option(SIMPLEMC_USE_STD_RANGES "Use ranges from the standard library" ${SIMPLEMC_USES_GNU})
option(SIMPLEMC_FETCH_FMT "Fetch fmt from github" ON)
option(SIMPLEMC_FETCH_RANGEV3 "Fetch range-v3 from github" ON)
option(SIMPLEMC_FETCH_EIGEN3 "Fetch Eigen3 from gitlab" ON)
option(SIMPLEMC_FETCH_NLOHMANN_JSON "Fetch nlohmann_json from github" ON)
option(SIMPLEMC_DISABLE_WARNINGS "Disable stricter compilation warnings" ON)

# output user options
message(STATUS "===============================================")
message(STATUS "Start configuring project ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Master project: ${SIMPLEMC_MASTER_PROJECT}")
message(STATUS "-----------------------------------------------")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Enable install: ${SIMPLEMC_ENABLE_INSTALL}")
message(STATUS "Build unit tests: ${SIMPLEMC_BUILD_TESTS}")
message(STATUS "Build examples: ${SIMPLEMC_BUILD_EXAMPLES}")
message(STATUS "Build documentation: ${SIMPLEMC_BUILD_DOC}")
message(STATUS "Use std ranges: ${SIMPLEMC_USE_STD_RANGES}")
message(STATUS "Fetch fmt: ${SIMPLEMC_FETCH_FMT}")
message(STATUS "Fetch range-v3: ${SIMPLEMC_FETCH_RANGEV3}")
message(STATUS "Fetch Eigen3: ${SIMPLEMC_FETCH_EIGEN3}")
message(STATUS "Fetch nlohmann_json: ${SIMPLEMC_FETCH_NLOHMANN_JSON}")
message(STATUS "Disable stricter compiler warnings: ${SIMPLEMC_DISABLE_WARNINGS}")
message(STATUS "-----------------------------------------------")

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
