# Build tests for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-test: Interface library to which other test targets link.
# tmp-test: Executable for testing temporary code.
# tmp-mpi-test: Executable for testing temporary MPI code.

# create interface library for testing
add_library(simplemc-test)
target_sources(simplemc-test PRIVATE test_utils.cpp)
target_include_directories(simplemc-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simplemc-test PUBLIC
    GTest::gtest_main
    simplemc::simplemc
    $<$<BOOL:${SIMPLEMC_USE_STD_RANGES}>:range-v3::range-v3>)

# library specific tests
add_subdirectory(utils)
add_subdirectory(mpi)
add_subdirectory(random)
add_subdirectory(grids)
add_subdirectory(numeric)
add_subdirectory(accs)

# temporary tests
add_executable(tmp-test tmp_test.cpp)
target_link_libraries(tmp-test PRIVATE simplemc-numeric)

add_executable(tmp-mpi-test tmp_mpi_test.cpp)
target_link_libraries(tmp-mpi-test PRIVATE simplemc::simplemc-mpi)
