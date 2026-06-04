# Build tests for simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# mc-simulation-run-test: Test simulation_params, simulation_stats, and simulation_run.
# mc-measurement-test: Test the type-erased measurement wrapper and the mc_measurement concept.

# create executables
add_executable(mc-simulation-run-test simulation_run_test.cpp)
target_link_libraries(mc-simulation-run-test PRIVATE simplemc-test)

add_executable(mc-measurement-test measurement_test.cpp)
target_link_libraries(mc-measurement-test PRIVATE simplemc-test)

# add tests
add_test(NAME mc-simulation-run-test COMMAND mc-simulation-run-test)
add_test(NAME mc-measurement-test COMMAND mc-measurement-test)
