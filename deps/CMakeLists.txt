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
        GIT_TAG origin/master)
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

# json
if(SIMPLEMC_FETCH_JSON)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG origin/develop)
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