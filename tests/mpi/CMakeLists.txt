# Build tests for simplemc-mpi.
#
# The following targets are defined:
# ---------------------------------
# mpi-utils-test: Executable to test the simplemc-mpi utility functions.
# mpi-env-comm-test: Executable to test the MPI environment and communicator wrappers.
# mpi-abort-exception-test: Executable to test the behavior of abort_on_exception == true.
# mpi-no-abort-exception-test: Executable to test the behavior of abort_on_exception == false.
# mpi-datatypes-test: Executable to test the mapping of MPI datatypes to C++ types.
# mpi-collectives-test: Executable to test the supported MPI collective communication operations.

add_executable(mpi-utils-test mpi_utils_test.cpp)
target_link_libraries(mpi-utils-test PRIVATE simplemc-test)
add_test(NAME mpi-utils-test COMMAND ./mpi-utils-test)

add_executable(mpi-env-comm-test mpi_env_comm_test.cpp)
target_link_libraries(mpi-env-comm-test PRIVATE simplemc-test)
add_test(NAME mpi-env-comm-test COMMAND mpirun -n 4 ./mpi-env-comm-test)

add_executable(mpi-abort-exception-test mpi_abort_exception_test.cpp)
target_link_libraries(mpi-abort-exception-test PRIVATE simplemc-test)

add_executable(mpi-no-abort-exception-test mpi_no_abort_exception_test.cpp)
target_link_libraries(mpi-no-abort-exception-test PRIVATE simplemc-test)

add_executable(mpi-datatypes-test mpi_datatypes_test.cpp)
target_link_libraries(mpi-datatypes-test PRIVATE simplemc-test)
add_test(NAME mpi-datatypes-test COMMAND ./mpi-datatypes-test)

add_executable(mpi-collectives-test mpi_collectives_test.cpp)
target_link_libraries(mpi-collectives-test PRIVATE simplemc-test)
add_test(NAME mpi-collectives-test COMMAND mpirun -n 4 ./mpi-collectives-test)
