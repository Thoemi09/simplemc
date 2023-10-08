include(FetchContent)

# googletest
if(SIMPLEMC_BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG main)
    set(INSTALL_GTEST OFF CACHE BOOL "")
    set(BUILD_GMOCK OFF CACHE BOOL "")
    message(STATUS "Fetching googletest ...")
    FetchContent_MakeAvailable(googletest)
    message(STATUS "googletest_SOURCE_DIR: ${googletest_SOURCE_DIR}")
endif()

# fmtlib
if(SIMPLEMC_FETCH_FMT)
    FetchContent_Declare(fmtlib
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG origin/master)
    message(STATUS "Fetching fmtlib ...")
    FetchContent_MakeAvailable(fmtlib)
    message(STATUS "fmtlib_SOURCE_DIR: ${fmtlib_SOURCE_DIR}")
else()
    message(STATUS "Finding fmt: fmt_DIR = ${fmt_DIR}")
    find_package(fmt REQUIRED)
    get_target_property(FMT_TMP fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "fmt found: Include directories = ${FMT_TMP}")
endif()