# Find/Fetch various dependencies of simplemc.
#
# The following dependencies are required:
# ---------------------------------
# fmt: Formatting library since std::format is not yet fully supported.
# range-v3: Range library since std::ranges is not yet fully supported (if SIMPLEMC_USE_STD_RANGES is
# ON it is only used for testing).
# mpi: MPI for parallelization.
# Eigen3: Linear algebra library.
# nlohmann_json: JSON parser for serialization and parsing configuration files.
#
# The following dependencies are optional:
# ---------------------------------
# googletest: For testing. Only used if SIMPLEMC_BUILD_TESTS is ON.

include(FetchContent)

# Helper function to safely fetch or find dependencies.
# This function checks if the target already exists before fetching or finding to avoid conflicts when
# simplemc is used as a subdirectory.
#
# Parameters:
#   TARGET_NAME: The name of the CMake target to check/create.
#   PACKAGE_NAME: The package name for find_package (defaults to TARGET_NAME if not provided).
#   SHOULD_FETCH: Boolean flag whether to fetch or find.
#   FETCH_CONTENT_NAME: Name for FetchContent operations.
#   GIT_REPO: Git repository URL.
#   GIT_TAG: Git tag/branch to fetch.
#   CACHE_OPTIONS: List of cache variables to set before calling FetchContent_MakeAvailable (format:
#                  "VAR1=VAL1;VAR2=VAL2;...").
function(simplemc_fetch_or_find_dependency)
    cmake_parse_arguments(
        ARG
        ""
        "TARGET_NAME;PACKAGE_NAME;SHOULD_FETCH;FETCH_CONTENT_NAME;GIT_REPO;GIT_TAG"
        "CACHE_OPTIONS;FIND_PACKAGE_ARGS"
        ${ARGN}
    )

    # default PACKAGE_NAME to TARGET_NAME if not provided
    if(NOT ARG_PACKAGE_NAME)
        set(ARG_PACKAGE_NAME ${ARG_TARGET_NAME})
    endif()

    # check if target already exists (from parent or previous configuration)
    if(NOT TARGET ${ARG_TARGET_NAME})
        if(ARG_SHOULD_FETCH)
            # check if already fetched/populated
            FetchContent_GetProperties(${ARG_FETCH_CONTENT_NAME})
            string(TOLOWER ${ARG_FETCH_CONTENT_NAME} fetch_name_lower)
            if(NOT ${fetch_name_lower}_POPULATED)
                # declare package to be fetched if it is not already populated
                FetchContent_Declare(${ARG_FETCH_CONTENT_NAME}
                    GIT_REPOSITORY ${ARG_GIT_REPO}
                    GIT_TAG ${ARG_GIT_TAG})

                # set cache options as INTERNAL to avoid polluting parent scope
                foreach(cache_opt ${ARG_CACHE_OPTIONS})
                    string(REPLACE "=" ";" opt_list ${cache_opt})
                    list(GET opt_list 0 opt_name)
                    list(GET opt_list 1 opt_value)
                    set(${opt_name} ${opt_value} CACHE INTERNAL "")
                endforeach()

                # fetch package
                message(STATUS "-----------------------------------------------")
                message(STATUS "Fetching ${ARG_PACKAGE_NAME} ...")
                FetchContent_MakeAvailable(${ARG_FETCH_CONTENT_NAME})
                message(STATUS "${ARG_FETCH_CONTENT_NAME}_SOURCE_DIR: ${${fetch_name_lower}_SOURCE_DIR}")
                message(STATUS "-----------------------------------------------")
            else()
                # skip fetch if it is already populated
                message(STATUS "-----------------------------------------------")
                message(STATUS "${ARG_PACKAGE_NAME} already populated, skipping fetch")
                message(STATUS "-----------------------------------------------")
            endif()
        else()
            # find package all
            message(STATUS "-----------------------------------------------")
            message(STATUS "Finding ${ARG_PACKAGE_NAME} ...")
            find_package(${ARG_PACKAGE_NAME} ${ARG_FIND_PACKAGE_ARGS} REQUIRED)

            # make imported target global to be visible in parent scope
            set_target_properties(${ARG_TARGET_NAME} PROPERTIES IMPORTED_GLOBAL TRUE)

            # output information about the package
            get_target_property(inc_dirs ${ARG_TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "${ARG_PACKAGE_NAME} found: Include directories = ${inc_dirs}")
            message(STATUS "-----------------------------------------------")
        endif()
    else()
        message(STATUS "-----------------------------------------------")
        message(STATUS "Target ${ARG_TARGET_NAME} already exists, skipping ${ARG_PACKAGE_NAME} fetch/find")
        message(STATUS "-----------------------------------------------")
    endif()
endfunction()

#########################
# Required dependencies
#########################

# fmtlib
simplemc_fetch_or_find_dependency(
    TARGET_NAME fmt::fmt
    PACKAGE_NAME fmt
    SHOULD_FETCH ${SIMPLEMC_FETCH_FMT}
    FETCH_CONTENT_NAME fmt
    GIT_REPO https://github.com/fmtlib/fmt
    GIT_TAG origin/master  # 0acf106c
    CACHE_OPTIONS "FMT_INSTALL=OFF"
)

# range-v3
if(NOT SIMPLEMC_USE_STD_RANGES)
    simplemc_fetch_or_find_dependency(
        TARGET_NAME range-v3::range-v3
        PACKAGE_NAME range-v3
        SHOULD_FETCH ${SIMPLEMC_FETCH_RANGEV3}
        FETCH_CONTENT_NAME rangev3
        GIT_REPO https://github.com/ericniebler/range-v3.git
        GIT_TAG origin/master  # 108f93c27
        CACHE_OPTIONS "RANGES_BUILD_CALENDAR_EXAMPLE=OFF"
    )
endif()

# mpi (always use find_package, never fetch)
simplemc_fetch_or_find_dependency(
    TARGET_NAME MPI::MPI_CXX
    PACKAGE_NAME MPI
    SHOULD_FETCH OFF
)

# Eigen
simplemc_fetch_or_find_dependency(
    TARGET_NAME Eigen3::Eigen
    PACKAGE_NAME Eigen3
    SHOULD_FETCH ${SIMPLEMC_FETCH_EIGEN3}
    FETCH_CONTENT_NAME eigen3
    GIT_REPO https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0  # fc0f148ab
    CACHE_OPTIONS "EIGEN_BUILD_TESTING=OFF"
)

# nlohmann_json
simplemc_fetch_or_find_dependency(
    TARGET_NAME nlohmann_json::nlohmann_json
    PACKAGE_NAME nlohmann_json
    SHOULD_FETCH ${SIMPLEMC_FETCH_NLOHMANN_JSON}
    FETCH_CONTENT_NAME nlohmann_json
    GIT_REPO https://github.com/nlohmann/json.git
    GIT_TAG origin/develop  # d10879bca
)

#########################
# Optional dependencies
#########################

# googletest (always fetch when needed, typically not provided by system)
if(SIMPLEMC_BUILD_TESTS)
    simplemc_fetch_or_find_dependency(
        TARGET_NAME GTest::gtest
        PACKAGE_NAME GTest
        SHOULD_FETCH ON
        FETCH_CONTENT_NAME googletest
        GIT_REPO https://github.com/google/googletest
        GIT_TAG main  # 8736d2cd
        CACHE_OPTIONS "INSTALL_GTEST=OFF;BUILD_GMOCK=OFF"
    )
endif()
