# Build tutorial codes that use simplemc::simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# tut_utils_1: Tutorial using simplemc::timer to measure runtime.
# tut_utils_2: Tutorial demonstrating multi-dimensional indexing.
# tut_utils_3: Tutorial using fmt to print formatted output.
# tut_mpi_1: Tutorial doing a "Hello, World!" using simplemc-mpi.
# tut_grids_1: Tutorial demonstating various 1-dimensional grids.
# tut_grids_2: Tutorial demonstating N-dimensional grids.

add_executable(tut_utils_1 tut_utils_1.cpp)
target_link_libraries(tut_utils_1 PRIVATE simplemc::simplemc)

add_executable(tut_utils_2 tut_utils_2.cpp)
target_link_libraries(tut_utils_2 PRIVATE simplemc::simplemc)

add_executable(tut_utils_3 tut_utils_3.cpp)
target_link_libraries(tut_utils_3 PRIVATE simplemc::simplemc)

add_executable(tut_mpi_1 tut_mpi_1.cpp)
target_link_libraries(tut_mpi_1 PRIVATE simplemc::simplemc)

add_executable(tut_grids_1 tut_grids_1.cpp)
target_link_libraries(tut_grids_1 PRIVATE simplemc::simplemc)

add_executable(tut_grids_2 tut_grids_2.cpp)
target_link_libraries(tut_grids_2 PRIVATE simplemc::simplemc)