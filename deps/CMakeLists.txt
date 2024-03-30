# Find/Fetch various dependencies of simplemc.
#
# The following dependencies are required:
# ---------------------------------
# fmtlib: It is fetched from github if SIMPLEMC_FETCH_FMT is ON, otherwise it is found using find_package.
# range-v3: It is fetched from github if SIMPLEMC_FETCH_RANGEV3 is ON, otherwise it is found using find_package.
# mpi: It is found using find_package.
# Eigen3: It is fetched from gitlab if SIMPLEMC_FETCH_EIGEN is ON, otherwise it is found using find_package.
# nlohmann_json: It is fetched from github if SIMPLEMC_FETCH_JSON is ON, otherwise it is found using find_package.
#
# The following dependencies are optional:
# ---------------------------------
# googletest: It is fetched from github and built only if SIMPLEMC_BUILD_TESTS is ON.

include(FetchContent)

#########################
# Required dependencies
#########################

# fmtlib
if(SIMPLEMC_FETCH_FMT)
    FetchContent_Declare(fmtlib
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG origin/master) # 130cf54
    set(FMT_INSTALL OFF CACHE BOOL "")
    message(STATUS "Fetching fmtlib ...")
    FetchContent_MakeAvailable(fmtlib)
    message(STATUS "fmtlib_SOURCE_DIR: ${fmtlib_SOURCE_DIR}")
else()
    message(STATUS "Finding fmt: fmt_DIR = ${fmt_DIR}")
    find_package(fmt REQUIRED)
    set_target_properties(fmt::fmt PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(FMT_TMP fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "fmt found: Include directories = ${FMT_TMP}")
endif()

# range-v3
if(SIMPLEMC_FETCH_RANGEV3)
    FetchContent_Declare(rangev3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
        GIT_TAG origin/master) # 9aa41d6
    set(RANGES_BUILD_CALENDAR_EXAMPLE OFF CACHE BOOL "")
    message(STATUS "Fetching range-v3 ...")
    FetchContent_MakeAvailable(rangev3)
    message(STATUS "rangev3_SOURCE_DIR: ${rangev3_SOURCE_DIR}")
else()
    message(STATUS "Finding range-v3: range-v3_DIR = ${range-v3_DIR}")
    find_package(range-v3 REQUIRED)
    set_target_properties(range-v3::range-v3 PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(RANGEV3_TMP range-v3::range-v3 INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "range-v3 found: Include directories = ${RANGEV3_TMP}")
endif()

# mpi
message(STATUS "Finding MPI ...")
find_package(MPI REQUIRED)
set_target_properties(MPI::MPI_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
message(STATUS "MPI found: Include directories = ${MPI_CXX_INCLUDE_DIRS}")

# Eigen
if(SIMPLEMC_FETCH_EIGEN)
    FetchContent_Declare(eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master) # aa6964b
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "")
    message(STATUS "Fetching eigen ...")
    FetchContent_MakeAvailable(eigen3)
    message(STATUS "eigen3_SOURCE_DIR: ${eigen3_SOURCE_DIR}")
else()
    message(STATUS "Finding Eigen3 ...")
    find_package(Eigen3 REQUIRED)
    set_target_properties(Eigen3::Eigen PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(EIGEN_TMP Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Eigen3 found: Include directories = ${EIGEN_TMP}")
endif()

# json
if(SIMPLEMC_FETCH_JSON)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG origin/develop) # edffad0
    message(STATUS "Fetching json ...")
    FetchContent_MakeAvailable(json)
    message(STATUS "json_SOURCE_DIR: ${json_SOURCE_DIR}")
else()
    message(STATUS "Finding json: nlohmann_json_DIR = ${nlohmann_json_DIR}")
    find_package(nlohmann_json REQUIRED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(JSON_TMP nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "json found: Include directories = ${JSON_TMP}")
endif()

#########################
# Optional dedendencies
#########################

# googletest
if(SIMPLEMC_BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG main) # 2dd1c13
    set(INSTALL_GTEST OFF CACHE BOOL "")
    set(BUILD_GMOCK OFF CACHE BOOL "")
    message(STATUS "Fetching googletest ...")
    FetchContent_MakeAvailable(googletest)
    message(STATUS "googletest_SOURCE_DIR: ${googletest_SOURCE_DIR}")
endif()
