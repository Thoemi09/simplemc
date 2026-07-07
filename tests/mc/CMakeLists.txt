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

add_executable(mc-update-set-test update_set_test.cpp)
target_link_libraries(mc-update-set-test PRIVATE simplemc-test)

add_executable(mc-kernel-test kernel_test.cpp)
target_link_libraries(mc-kernel-test PRIVATE simplemc-test)

add_executable(mc-metropolis-kernel-test metropolis_kernel_test.cpp)
target_link_libraries(mc-metropolis-kernel-test PRIVATE simplemc-test)

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
add_test(NAME mc-progress-printer-test COMMAND mc-progress-printer-test)
add_test(NAME mc-mpi-test COMMAND mpirun -n 4 mc-mpi-test)
