# Build tutorial codes that use simplemc::simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# tut_utils_1: Using simplemc::timer to measure runtime.
# tut_utils_2: Multi-dimensional indexing.
# tut_utils_3: Using fmt to print formatted output.
# tut_mpi_1: "Hello, World!" using simplemc-mpi.
# tut_mpi_2_1: Error handling: Working example that broadcasts a simple message.
# tut_mpi_2_2: Error handling: Using an invalid root rank.
# tut_mpi_2_3: Error handling: Setting the MPI error handler to return errors.
# tut_mpi_2_4: Error handling: Catch exceptions locally.
# tut_mpi_2_5: Error handling: Abort on exception set to true.
# tut_mpi_2_6: Error handling: Abort on exception set to false.
# tut_mpi_3: Common collective MPI operations.
# tut_grids_1: Various 1-dimensional grids.
# tut_grids_2: N-dimensional grids.

add_executable(tut_utils_1 tut_utils_1.cpp)
target_link_libraries(tut_utils_1 PRIVATE simplemc::simplemc)

add_executable(tut_utils_2 tut_utils_2.cpp)
target_link_libraries(tut_utils_2 PRIVATE simplemc::simplemc)

add_executable(tut_utils_3 tut_utils_3.cpp)
target_link_libraries(tut_utils_3 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_1 tut_mpi_1.cpp)
target_link_libraries(tut_mpi_1 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_2_1 tut_mpi_2_1.cpp)
target_link_libraries(tut_mpi_2_1 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_2_2 tut_mpi_2_2.cpp)
target_link_libraries(tut_mpi_2_2 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_2_3 tut_mpi_2_3.cpp)
target_link_libraries(tut_mpi_2_3 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_2_4 tut_mpi_2_4.cpp)
target_link_libraries(tut_mpi_2_4 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_2_5 tut_mpi_2_5.cpp)
target_link_libraries(tut_mpi_2_5 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_2_6 tut_mpi_2_6.cpp)
target_link_libraries(tut_mpi_2_6 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_3 tut_mpi_3.cpp)
target_link_libraries(tut_mpi_3 PRIVATE simplemc::simplemc)

add_executable(tut_grids_1 tut_grids_1.cpp)
target_link_libraries(tut_grids_1 PRIVATE simplemc::simplemc)

add_executable(tut_grids_2 tut_grids_2.cpp)
target_link_libraries(tut_grids_2 PRIVATE simplemc::simplemc)
