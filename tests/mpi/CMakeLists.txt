add_executable(mpi-test mpi_test.cpp)
target_link_libraries(mpi-test PRIVATE simplemc::simplemc-mpi simplemc-test)
add_test(NAME mpi-test COMMAND mpirun -n 4 ./mpi-test)

add_executable(env-comm-test env_comm_test.cpp)
target_link_libraries(env-comm-test PRIVATE simplemc::simplemc-mpi simplemc-test)
add_test(NAME env-comm-test COMMAND mpirun -n 4 ./env-comm-test)
