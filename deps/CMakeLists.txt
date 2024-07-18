# Find/Fetch various dependencies of simplemc.
#
# The following dependencies are required:
# ---------------------------------
# fmt: Formatting library since std::format is not yet fully supported.
# range-v3: Range library since std::ranges is not yet fully supported.
# mpi: MPI for parallelization.
# Eigen3: Linear algebra library.
# nlohmann_json: JSON parser for serialization and configuration.
#
# The following dependencies are optional:
# ---------------------------------
# googletest: For testing. Only used if SIMPLEMC_BUILD_TESTS is ON.
# HighFive: HDF5 wrapper for serialization. Only used if SIMPLEMC_WITH_HIGHFIVE is ON.

include(FetchContent)

#########################
# Required dependencies
#########################

# fmtlib
if(SIMPLEMC_FETCH_FMT)
    FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG origin/master) # 130cf54
    set(FMT_INSTALL OFF CACHE BOOL "")
    message(STATUS "-----------------------------------------------")
    message(STATUS "Fetching fmt ...")
    FetchContent_MakeAvailable(fmt)
    message(STATUS "fmt_SOURCE_DIR: ${fmt_SOURCE_DIR}")
    message(STATUS "-----------------------------------------------")
else()
    message(STATUS "-----------------------------------------------")
    message(STATUS "Finding fmtlib: fmt_DIR = ${fmt_DIR}")
    find_package(fmt REQUIRED)
    set_target_properties(fmt::fmt PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(FMT_TMP fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "fmt found: Include directories = ${FMT_TMP}")
    message(STATUS "-----------------------------------------------")
endif()

# range-v3
if(SIMPLEMC_FETCH_RANGEV3)
    FetchContent_Declare(rangev3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
        GIT_TAG origin/master) # 9aa41d6
    set(RANGES_BUILD_CALENDAR_EXAMPLE OFF CACHE BOOL "")
    message(STATUS "-----------------------------------------------")
    message(STATUS "Fetching range-v3 ...")
    FetchContent_MakeAvailable(rangev3)
    message(STATUS "rangev3_SOURCE_DIR: ${rangev3_SOURCE_DIR}")
    message(STATUS "-----------------------------------------------")
else()
message(STATUS "-----------------------------------------------")
    message(STATUS "Finding range-v3: range-v3_DIR = ${range-v3_DIR}")
    find_package(range-v3 REQUIRED)
    set_target_properties(range-v3::range-v3 PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(RANGEV3_TMP range-v3 INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "range-v3 found: Include directories = ${RANGEV3_TMP}")
    message(STATUS "-----------------------------------------------")
endif()

# mpi
message(STATUS "-----------------------------------------------")
message(STATUS "Finding MPI ...")
find_package(MPI REQUIRED)
set_target_properties(MPI::MPI_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
message(STATUS "MPI found: Include directories = ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "-----------------------------------------------")

# Eigen
if(SIMPLEMC_FETCH_EIGEN3)
    FetchContent_Declare(eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master) # aa6964b
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "")
    message(STATUS "-----------------------------------------------")
    message(STATUS "Fetching Eigen3 ...")
    FetchContent_MakeAvailable(eigen3)
    message(STATUS "eigen3_SOURCE_DIR: ${eigen3_SOURCE_DIR}")
    message(STATUS "-----------------------------------------------")
else()
    message(STATUS "-----------------------------------------------")
    message(STATUS "Finding Eigen3 ...")
    find_package(Eigen3 REQUIRED)
    set_target_properties(Eigen3::Eigen PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(EIGEN3_TMP Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Eigen3 found: Include directories = ${EIGEN3_TMP}")
    message(STATUS "-----------------------------------------------")
endif()

# nlohmann_json
if(SIMPLEMC_FETCH_NLOHMANN_JSON)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG origin/develop) # edffad0
    message(STATUS "-----------------------------------------------")
    message(STATUS "Fetching nlohmann_json ...")
    FetchContent_MakeAvailable(nlohmann_json)
    message(STATUS "nlohmann_json_SOURCE_DIR: ${nlohmann_json_SOURCE_DIR}")
    message(STATUS "-----------------------------------------------")
else()
    message(STATUS "-----------------------------------------------")
    message(STATUS "Finding nlohmann_json: nlohmann_json_DIR = ${nlohmann_json_DIR}")
    find_package(nlohmann_json REQUIRED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES IMPORTED_GLOBAL TRUE)
    get_target_property(NLOHMANN_JSON_TMP nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "nlohmann_json found: Include directories = ${NLOHMANN_JSON_TMP}")
    message(STATUS "-----------------------------------------------")
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
    message(STATUS "-----------------------------------------------")
    message(STATUS "Fetching googletest ...")
    FetchContent_MakeAvailable(googletest)
    message(STATUS "googletest_SOURCE_DIR: ${googletest_SOURCE_DIR}")
    message(STATUS "-----------------------------------------------")
endif()

# HighFive
if(SIMPLEMC_WITH_HIGHFIVE)
    if(SIMPLEMC_FETCH_HIGHFIVE)
        FetchContent_Declare(highfive
            GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
            GIT_TAG origin/master)
        message(STATUS "-----------------------------------------------")
        message(STATUS "Fetching highfive ...")
        FetchContent_MakeAvailable(HighFive)
        message(STATUS "highfive_SOURCE_DIR: ${highfive_SOURCE_DIR}")
        message(STATUS "-----------------------------------------------")
    else()
        message(STATUS "-----------------------------------------------")
        message(STATUS "Finding HighFive: HighFive_DIR = ${HighFive_DIR}")
        find_package(HighFive REQUIRED)
        set_target_properties(HighFive::HighFive PROPERTIES IMPORTED_GLOBAL TRUE)
        get_target_property(HIGHFIVE_TMP HighFive::Include INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "HighFive found: Include directories = ${HIGHFIVE_TMP}")
        message(STATUS "-----------------------------------------------")
    endif()
endif()
