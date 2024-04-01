# Build tests for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-test: Interface library to which other test targets link.
# tmp-test: Executable for testing temporary code.
# tmp-mpi-test: Executable for testing temporary MPI code.

# create interface library for testing
add_library(simplemc-test INTERFACE)
target_include_directories(simplemc-test INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simplemc-test INTERFACE GTest::gtest_main simplemc::simplemc-utils)

# library specific tests
add_subdirectory(utils)
add_subdirectory(json)
add_subdirectory(random)
add_subdirectory(grids)
add_subdirectory(mpi)
add_subdirectory(numeric)
add_subdirectory(accs)

# temporary tests
add_executable(tmp-test tmp_test.cpp)
target_link_libraries(tmp-test PRIVATE simplemc-test simplemc::simplemc-accs)

add_executable(tmp-mpi-test tmp_mpi_test.cpp)
target_link_libraries(tmp-mpi-test PRIVATE simplemc::simplemc-mpi)
