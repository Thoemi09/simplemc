# Build stochastic process tests for simplemc-accs.
#
## The following targets are defined:
# ---------------------------------
#
# accs-stochastic-process-utils-test: Test the utilities.
# accs-stochastic-process-mean-acc-test: Test the mean accumulator.
# accs-stochastic-process-var-acc-test: Test the variance accumulator.
# accs-stochastic-process-covar-acc-test: Test the covariance accumulator.
# accs-stochastic-process-batch-acc-test: Test the batch accumulator.
# accs-stochastic-process-mpi-test: Test the MPI support.

add_executable(accs-stochastic-process-utils-test stochastic_process_utils_test.cpp)
target_link_libraries(accs-stochastic-process-utils-test PRIVATE simplemc-test)

add_executable(accs-stochastic-process-mean-acc-test stochastic_process_mean_acc_test.cpp)
target_link_libraries(accs-stochastic-process-mean-acc-test PRIVATE simplemc-test)

add_executable(accs-stochastic-process-var-acc-test stochastic_process_var_acc_test.cpp)
target_link_libraries(accs-stochastic-process-var-acc-test PRIVATE simplemc-test)

add_executable(accs-stochastic-process-covar-acc-test stochastic_process_covar_acc_test.cpp)
target_link_libraries(accs-stochastic-process-covar-acc-test PRIVATE simplemc-test)

add_executable(accs-stochastic-process-batch-acc-test stochastic_process_batch_acc_test.cpp)
target_link_libraries(accs-stochastic-process-batch-acc-test PRIVATE simplemc-test)

add_executable(accs-stochastic-process-mpi-test stochastic_process_mpi_test.cpp)
target_link_libraries(accs-stochastic-process-mpi-test PRIVATE simplemc-test)

add_test(NAME accs-stochastic-process-utils-test COMMAND accs-stochastic-process-utils-test)
add_test(NAME accs-stochastic-process-mean-acc-test COMMAND accs-stochastic-process-mean-acc-test)
add_test(NAME accs-stochastic-process-var-acc-test COMMAND accs-stochastic-process-var-acc-test)
add_test(NAME accs-stochastic-process-covar-acc-test COMMAND accs-stochastic-process-covar-acc-test)
add_test(NAME accs-stochastic-process-batch-acc-test COMMAND accs-stochastic-process-batch-acc-test)
add_test(NAME accs-stochastic-process-mpi-test COMMAND mpirun -n 4 accs-stochastic-process-mpi-test)
