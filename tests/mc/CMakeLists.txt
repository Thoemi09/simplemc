# Build tests for simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# mc-simulation-run-test: Test simulation_params, simulation_stats, and simulation_run.

# create executables
add_executable(mc-simulation-run-test simulation_run_test.cpp)
target_link_libraries(mc-simulation-run-test PRIVATE simplemc-test)

# add tests
add_test(NAME mc-simulation-run-test COMMAND mc-simulation-run-test)
