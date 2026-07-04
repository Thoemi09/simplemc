# Build tests for simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# mc-tuple-set-test: Test the tuple_set base (traversal, lookup, typed access).
# mc-stats-and-params-test: Test simulation_params and simulation_stats.
# mc-measurement-test: Test measurement<M> wrapper + mc_measurement concept.
# mc-measurement-set-test: Test measurement_set (construction, active cache, measure_all, serialization).
# mc-update-test: Test update<U> wrapper + mc_update concept.
# mc-update-set-test: Test update_set (construction, link_pair, weight, select, serialization).
# mc-kernel-test: Test the mc_kernel concept.
# mc-metropolis-kernel-test: Test the default Metropolis step kernel.
# mc-run-test: Test the free run() driver + run_callbacks + custom kernels.
# mc-checkpoint-test: Test the free save/load helpers over the run components.
# mc-input-config-test: Test input-config serialization of params + run components.
# mc-mixed-backend-test: Test that the checkpoint and input-config channels stay independent.
# mc-callbacks-test: Test progress_printer and the unified checkpoint API (writer + save/load + deduction).
# mc-mpi-test: Test MPI reduction helpers on simplemc-mc types (mpirun -n 4).

# create executables
add_executable(mc-tuple-set-test tuple_set_test.cpp)
target_link_libraries(mc-tuple-set-test PRIVATE simplemc-test)

add_executable(mc-stats-and-params-test stats_and_params_test.cpp)
target_link_libraries(mc-stats-and-params-test PRIVATE simplemc-test)

add_executable(mc-measurement-test measurement_test.cpp)
target_link_libraries(mc-measurement-test PRIVATE simplemc-test)

add_executable(mc-measurement-set-test measurement_set_test.cpp)
target_link_libraries(mc-measurement-set-test PRIVATE simplemc-test)

add_executable(mc-update-test update_test.cpp)
target_link_libraries(mc-update-test PRIVATE simplemc-test)

add_executable(mc-update-set-test update_set_test.cpp)
target_link_libraries(mc-update-set-test PRIVATE simplemc-test)

add_executable(mc-kernel-test kernel_test.cpp)
target_link_libraries(mc-kernel-test PRIVATE simplemc-test)

add_executable(mc-metropolis-kernel-test metropolis_kernel_test.cpp)
target_link_libraries(mc-metropolis-kernel-test PRIVATE simplemc-test)

add_executable(mc-run-test run_test.cpp)
target_link_libraries(mc-run-test PRIVATE simplemc-test)

add_executable(mc-checkpoint-test checkpoint_test.cpp)
target_link_libraries(mc-checkpoint-test PRIVATE simplemc-test)

add_executable(mc-input-config-test input_config_test.cpp)
target_link_libraries(mc-input-config-test PRIVATE simplemc-test)

add_executable(mc-mixed-backend-test mixed_backend_test.cpp)
target_link_libraries(mc-mixed-backend-test PRIVATE simplemc-test)

add_executable(mc-callbacks-test callbacks_test.cpp)
target_link_libraries(mc-callbacks-test PRIVATE simplemc-test)

add_executable(mc-mpi-test mc_mpi_test.cpp)
target_link_libraries(mc-mpi-test PRIVATE simplemc-test)

# add tests
add_test(NAME mc-tuple-set-test COMMAND mc-tuple-set-test)
add_test(NAME mc-stats-and-params-test COMMAND mc-stats-and-params-test)
add_test(NAME mc-measurement-test COMMAND mc-measurement-test)
add_test(NAME mc-measurement-set-test COMMAND mc-measurement-set-test)
add_test(NAME mc-update-test COMMAND mc-update-test)
add_test(NAME mc-update-set-test COMMAND mc-update-set-test)
add_test(NAME mc-kernel-test COMMAND mc-kernel-test)
add_test(NAME mc-metropolis-kernel-test COMMAND mc-metropolis-kernel-test)
add_test(NAME mc-run-test COMMAND mc-run-test)
add_test(NAME mc-checkpoint-test COMMAND mc-checkpoint-test)
add_test(NAME mc-input-config-test COMMAND mc-input-config-test)
add_test(NAME mc-mixed-backend-test COMMAND mc-mixed-backend-test)
add_test(NAME mc-callbacks-test COMMAND mc-callbacks-test)
add_test(NAME mc-mpi-test COMMAND mpirun -n 4 mc-mpi-test)
