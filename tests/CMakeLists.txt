# Build tests for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-test

include(UnitTests)

# create interface library for testing
add_library(simplemc-test INTERFACE)
target_include_directories(simplemc-test INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simplemc-test INTERFACE GTest::gtest_main simplemc::simplemc-json)

# library specific tests
add_subdirectory(utils)
add_subdirectory(json)
add_subdirectory(random)

# temporary test
add_unittest(NAME tmp-test SOURCES tmp_test.cpp LIBS simplemc::simplemc-random)
