# Build tests for simplemc-mpi.
#
# The following targets are defined:
# ---------------------------------
# mpi-utils-test: Executable to test the simplemc-mpi utility functions.
# mpi-env-test: Executable to test the MPI environment wrappers.
# mpi-group-test: Executable to test the MPI group wrapper.
# mpi-comm-test: Executable to test the MPI communicator wrapper.
# mpi-datatype-test: Executable to test the MPI datatype wrapper.
# mpi-abort-exception-test: Executable to test the behavior of abort_on_exception == true.
# mpi-no-abort-exception-test: Executable to test the behavior of abort_on_exception == false.
# mpi-datatypes-test: Executable to test the mapping of MPI datatypes to C++ types.
# mpi-collectives-test: Executable to test the supported MPI collective communication operations.
# mpi-threading-test: Executable to test MPI threading support functions.

# create executables
add_executable(mpi-utils-test mpi_utils_test.cpp)
target_link_libraries(mpi-utils-test PRIVATE simplemc-test)

add_executable(mpi-env-test mpi_env_test.cpp)
target_link_libraries(mpi-env-test PRIVATE simplemc-test)

add_executable(mpi-group-test mpi_group_test.cpp)
target_link_libraries(mpi-group-test PRIVATE simplemc-test)

add_executable(mpi-comm-test mpi_comm_test.cpp)
target_link_libraries(mpi-comm-test PRIVATE simplemc-test)

add_executable(mpi-datatype-test mpi_datatype_test.cpp)
target_link_libraries(mpi-datatype-test PRIVATE simplemc-test)

add_executable(mpi-abort-exception-test mpi_abort_exception_test.cpp)
target_link_libraries(mpi-abort-exception-test PRIVATE simplemc-test)

add_executable(mpi-no-abort-exception-test mpi_no_abort_exception_test.cpp)
target_link_libraries(mpi-no-abort-exception-test PRIVATE simplemc-test)

add_executable(mpi-datatypes-test mpi_datatypes_test.cpp)
target_link_libraries(mpi-datatypes-test PRIVATE simplemc-test)

add_executable(mpi-collectives-test mpi_collectives_test.cpp)
target_link_libraries(mpi-collectives-test PRIVATE simplemc-test)

add_executable(mpi-threading-test mpi_threading_test.cpp)
target_link_libraries(mpi-threading-test PRIVATE simplemc-test)

# add tests
add_test(NAME mpi-utils-test COMMAND ./mpi-utils-test)
add_test(NAME mpi-env-test COMMAND mpirun -n 4 ./mpi-env-test)
add_test(NAME mpi-group-test COMMAND mpirun -n 4 ./mpi-group-test)
add_test(NAME mpi-comm-test COMMAND mpirun -n 4 ./mpi-comm-test)
add_test(NAME mpi-datatype-test COMMAND mpirun -n 4 ./mpi-datatype-test)
add_test(NAME mpi-datatypes-test COMMAND ./mpi-datatypes-test)
add_test(NAME mpi-collectives-test COMMAND mpirun -n 4 ./mpi-collectives-test)
add_test(NAME mpi-threading-test COMMAND mpirun -n 4 ./mpi-threading-test)
