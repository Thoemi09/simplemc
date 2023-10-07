include(UnitTests)
include(FetchContent)

# fetch googletest
if(SIMPLEMC_BUILD_TESTS)
    message(STATUS "Fetching googletest ...")
    set(INSTALL_GTEST OFF CACHE BOOL "")
    set(BUILD_GMOCK OFF CACHE BOOL "")
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG main)
    FetchContent_MakeAvailable(googletest)
    message(STATUS "googletest_SOURCE_DIR: ${googletest_SOURCE_DIR}")
endif()
set(GTEST_LIB $<IF:$<TARGET_EXISTS:GTest::Main>,GTest::Main,gtest_main>)

# temporary test (should be disabled or removed at the end)
add_unittest(NAME tmp-test SOURCES tmp_test.cpp)
