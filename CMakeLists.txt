# User options/variables:
# ------------------------
# SIMPLEMC_ENABLE_INSTALL: Enable install target.
# SIMPLEMC_BUILD_TESTS: Build unit tests with googletest.
# SIMPLEMC_BUILD_EXAMPLES: Build example code.
# SIMPLEMC_FETCH_JSON: Fetch nlohmann_json library from github.
# SIMPLEMC_FETCH_FMT: Fetch fmtlib library from github.
# SIMPLEMC_FETCH_EIGEN: Fetch eigen library from gitlab.
# SIMPLEMC_DISABLE_WARNINGS: Stricter compilation flags.

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
option(SIMPLEMC_FETCH_JSON "Fetch nlohmann_json from github" ON)
option(SIMPLEMC_FETCH_FMT "Fetch fmtlib from github" ON)
option(SIMPLEMC_FETCH_EIGEN "Fetch eigen from gitlab" ON)
option(SIMPLEMC_DISABLE_WARNINGS "Disable stricter compilation flags" OFF)

message(STATUS "Start configuring project ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Enable install: ${SIMPLEMC_ENABLE_INSTALL}")
message(STATUS "Build unit tests: ${SIMPLEMC_BUILD_TESTS}")
message(STATUS "Build examples: ${SIMPLEMC_BUILD_EXAMPLES}")
message(STATUS "Fetch nlohmann-json: ${SIMPLEMC_FETCH_JSON}")
message(STATUS "Fetch fmtlib: ${SIMPLEMC_FETCH_FMT}")
message(STATUS "Fetch eigen: ${SIMPLEMC_FETCH_EIGEN}")
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