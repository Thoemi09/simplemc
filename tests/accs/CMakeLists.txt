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
# Stochastic process tests (SIMPLEMC_BUILD_STOCHASTIC_PROCESS_TESTS=ON):
# See stochastic_process/CMakeLists.txt for targets.

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

add_executable(accs-jackknife-test accs_jackknife_test.cpp)
target_link_libraries(accs-jackknife-test PRIVATE simplemc-test)

add_executable(accs-mpi-test accs_mpi_test.cpp)
target_link_libraries(accs-mpi-test PRIVATE simplemc-test)

add_test(NAME accs-utils-test COMMAND accs-utils-test)
add_test(NAME accs-mean-acc-test COMMAND accs-mean-acc-test)
add_test(NAME accs-var-acc-test COMMAND accs-var-acc-test)
add_test(NAME accs-covar-acc-test COMMAND accs-covar-acc-test)
add_test(NAME accs-batch-acc-test COMMAND accs-batch-acc-test)
add_test(NAME accs-jackknife-test COMMAND accs-jackknife-test)
add_test(NAME accs-mpi-test COMMAND mpirun -n 4 accs-mpi-test)

# stochastic process based tests
if(SIMPLEMC_BUILD_STOCHASTIC_PROCESS_TESTS)
    add_subdirectory(stochastic_process)
endif()
