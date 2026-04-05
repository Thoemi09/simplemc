# Build tests for simplemc-accs.
#
## The following targets are defined:
# ---------------------------------
#
# Deterministic tests (always built):
# accs-utils-test: Test the utilities of the simplemc-accs library.
# accs-mean-acc-test: Test the mean accumulator.
# accs-var-acc-test: Test the variance accumulator.
# accs-covar-acc-test: Test the covariance accumulator.
# accs-batch-acc-test: Test the batch accumulator.
# accs-mpi-test: Test the MPI support.
#
# Advanced tests using a stochastic process (SIMPLEMC_BUILD_ADVANCED_TESTS=ON):
# accs-utils-advanced-test: Advanced utils tests.
# accs-mean-acc-advanced-test: Advanced mean accumulator tests.
# accs-var-acc-advanced-test: Advanced variance accumulator tests.
# accs-covar-acc-advanced-test: Advanced covariance accumulator tests.
# accs-batch-acc-advanced-test: Advanced batch accumulator tests.
# accs-mpi-advanced-test: Advanced MPI tests.

# Deterministic tests (fast, small data sets)

add_executable(accs-utils-test accs_utils_test.cpp)
target_link_libraries(accs-utils-test PRIVATE simplemc-test)

add_executable(accs-mean-acc-test accs_mean_acc_test.cpp)
target_link_libraries(accs-mean-acc-test PRIVATE simplemc-test)

add_executable(accs-var-acc-test accs_var_acc_test.cpp)
target_link_libraries(accs-var-acc-test PRIVATE simplemc-test)

add_executable(accs-covar-acc-test accs_covar_acc_test.cpp)
target_link_libraries(accs-covar-acc-test PRIVATE simplemc-test)

add_executable(accs-batch-acc-test accs_batch_acc_test.cpp)
target_link_libraries(accs-batch-acc-test PRIVATE simplemc-test)

add_executable(accs-mpi-test accs_mpi_test.cpp)
target_link_libraries(accs-mpi-test PRIVATE simplemc-test)

add_test(NAME accs-utils-test COMMAND accs-utils-test)
add_test(NAME accs-mean-acc-test COMMAND accs-mean-acc-test)
add_test(NAME accs-var-acc-test COMMAND accs-var-acc-test)
add_test(NAME accs-covar-acc-test COMMAND accs-covar-acc-test)
add_test(NAME accs-batch-acc-test COMMAND accs-batch-acc-test)
add_test(NAME accs-mpi-test COMMAND mpirun -n 4 accs-mpi-test)

# Advanced tests (stochastic process based, gated by option) 
if(SIMPLEMC_BUILD_ADVANCED_TESTS)
    add_executable(accs-utils-advanced-test accs_utils_advanced_test.cpp)
    target_link_libraries(accs-utils-advanced-test PRIVATE simplemc-test)

    add_executable(accs-mean-acc-advanced-test accs_mean_acc_advanced_test.cpp)
    target_link_libraries(accs-mean-acc-advanced-test PRIVATE simplemc-test)

    add_executable(accs-var-acc-advanced-test accs_var_acc_advanced_test.cpp)
    target_link_libraries(accs-var-acc-advanced-test PRIVATE simplemc-test)

    add_executable(accs-covar-acc-advanced-test accs_covar_acc_advanced_test.cpp)
    target_link_libraries(accs-covar-acc-advanced-test PRIVATE simplemc-test)

    add_executable(accs-batch-acc-advanced-test accs_batch_acc_advanced_test.cpp)
    target_link_libraries(accs-batch-acc-advanced-test PRIVATE simplemc-test)

    add_executable(accs-mpi-advanced-test accs_mpi_advanced_test.cpp)
    target_link_libraries(accs-mpi-advanced-test PRIVATE simplemc-test)

    add_test(NAME accs-utils-advanced-test COMMAND accs-utils-advanced-test)
    add_test(NAME accs-mean-acc-advanced-test COMMAND accs-mean-acc-advanced-test)
    add_test(NAME accs-var-acc-advanced-test COMMAND accs-var-acc-advanced-test)
    add_test(NAME accs-covar-acc-advanced-test COMMAND accs-covar-acc-advanced-test)
    add_test(NAME accs-batch-acc-advanced-test COMMAND accs-batch-acc-advanced-test)
    add_test(NAME accs-mpi-advanced-test COMMAND mpirun -n 4 accs-mpi-advanced-test)
endif()
