# Build tests for simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# mc-tuple-set-test: Test the tuple_set base class (traversal, lookup, typed access).
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
# mc-progress-printer-test: Test progress_printer and progress_line.
# mc-mpi-test: Test MPI reduction helpers on simplemc-mc types (mpirun -n 4).

# create executables
add_executable(mc-tuple-set-test mc_tuple_set_test.cpp)
target_link_libraries(mc-tuple-set-test PRIVATE simplemc-test)

add_executable(mc-stats-and-params-test mc_stats_and_params_test.cpp)
target_link_libraries(mc-stats-and-params-test PRIVATE simplemc-test)

add_executable(mc-measurement-test mc_measurement_test.cpp)
target_link_libraries(mc-measurement-test PRIVATE simplemc-test)

add_executable(mc-measurement-set-test mc_measurement_set_test.cpp)
target_link_libraries(mc-measurement-set-test PRIVATE simplemc-test)

add_executable(mc-update-test mc_update_test.cpp)
target_link_libraries(mc-update-test PRIVATE simplemc-test)

add_executable(mc-update-set-test mc_update_set_test.cpp)
target_link_libraries(mc-update-set-test PRIVATE simplemc-test)

add_executable(mc-kernel-test mc_kernel_test.cpp)
target_link_libraries(mc-kernel-test PRIVATE simplemc-test)

add_executable(mc-metropolis-kernel-test mc_metropolis_kernel_test.cpp)
target_link_libraries(mc-metropolis-kernel-test PRIVATE simplemc-test)

add_executable(mc-run-test mc_run_test.cpp)
target_link_libraries(mc-run-test PRIVATE simplemc-test)

add_executable(mc-checkpoint-test mc_checkpoint_test.cpp)
target_link_libraries(mc-checkpoint-test PRIVATE simplemc-test)

add_executable(mc-input-config-test mc_input_config_test.cpp)
target_link_libraries(mc-input-config-test PRIVATE simplemc-test)

add_executable(mc-progress-printer-test mc_progress_printer_test.cpp)
target_link_libraries(mc-progress-printer-test PRIVATE simplemc-test)

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
add_test(NAME mc-progress-printer-test COMMAND mc-progress-printer-test)
add_test(NAME mc-mpi-test COMMAND mpirun -n 4 mc-mpi-test)
