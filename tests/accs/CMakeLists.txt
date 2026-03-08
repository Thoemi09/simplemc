# Build tests for simplemc-accs.
#
# The following targets are defined:
# ---------------------------------
# accs-utils-test: Executable to test the utilities of the simplemc-accs library.
# accs-mean-acc-test: Executable to test the mean accumulator of the simplemc-accs library.
# accs-var-acc-test: Executable to test the variance accumulator of the simplemc-accs library.
# accs-covar-acc-test: Executable to test the covariance accumulator of the simplemc-accs library.
# accs-batch-acc-test: Executable to test the batch accumulator of the simplemc-accs library.
# accs-mpi-test: Executable to test the MPI support of the simplemc-accs library.

# create executables
add_executable(accs-utils-test accs_utils_test.cpp)
target_link_libraries(accs-utils-test PRIVATE simplemc-test)

add_executable(accs-mean-acc-test accs_mean_acc_test.cpp)
target_link_libraries(accs-mean-acc-test PRIVATE simplemc-test)

# FIXME: Temporarily disabled during accs restructure.
# add_executable(accs-var-acc-test accs_var_acc_test.cpp)
# target_link_libraries(accs-var-acc-test PRIVATE simplemc-test)

# add_executable(accs-covar-acc-test accs_covar_acc_test.cpp)
# target_link_libraries(accs-covar-acc-test PRIVATE simplemc-test)

# add_executable(accs-batch-acc-test accs_batch_acc_test.cpp)
# target_link_libraries(accs-batch-acc-test PRIVATE simplemc-test)

# add_executable(accs-mpi-test accs_mpi_test.cpp)
# target_link_libraries(accs-mpi-test PRIVATE simplemc-test)

# add tests
add_test(NAME accs-utils-test COMMAND accs-utils-test)
add_test(NAME accs-mean-acc-test COMMAND accs-mean-acc-test)
# FIXME: Temporarily disabled during accs restructure.
# add_test(NAME accs-var-acc-test COMMAND accs-var-acc-test)
# add_test(NAME accs-covar-acc-test COMMAND accs-covar-acc-test)
# add_test(NAME accs-batch-acc-test COMMAND accs-batch-acc-test)
# add_test(NAME accs-mpi-test COMMAND mpirun -n 4 accs-mpi-test)
