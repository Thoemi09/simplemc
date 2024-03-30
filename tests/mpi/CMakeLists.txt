# Build tests for simplemc-mpi.
#
# The following targets are defined:
# ---------------------------------
# mpi-test: Executable to test the simplemc-mpi library.
# env-comm-test: Executable to test the MPI environment and communicator wrappers.
# abort-exception-test: Executable to test the behavior of abort_on_exception == true.
# no-abort-exception-test: Executable to test the behavior of abort_on_exception == false.

add_executable(mpi-test mpi_test.cpp)
target_link_libraries(mpi-test PRIVATE simplemc::simplemc-mpi simplemc-test)
add_test(NAME mpi-test COMMAND mpirun -n 4 ./mpi-test)

add_executable(env-comm-test env_comm_test.cpp)
target_link_libraries(env-comm-test PRIVATE simplemc::simplemc-mpi simplemc-test)
add_test(NAME env-comm-test COMMAND mpirun -n 4 ./env-comm-test)

add_executable(abort-exception-test abort_exception_test.cpp)
target_link_libraries(abort-exception-test PRIVATE simplemc::simplemc-mpi simplemc-test)

add_executable(no-abort-exception-test no_abort_exception_test.cpp)
target_link_libraries(no-abort-exception-test PRIVATE simplemc::simplemc-mpi simplemc-test)
