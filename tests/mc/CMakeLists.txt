# Build tests for simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# mc-stats-and-params-test: Test simulation_params, simulation_stats, and update_stats.
# mc-measurement-test: Test the type-erased measurement wrapper and the mc_measurement concept.
# mc-update-test: Test the type-erased update wrapper and the mc_update concept.
# mc-simulation-test: Test the simulation run-loop driver.

# create executables
add_executable(mc-stats-and-params-test stats_and_params_test.cpp)
target_link_libraries(mc-stats-and-params-test PRIVATE simplemc-test)

add_executable(mc-measurement-test measurement_test.cpp)
target_link_libraries(mc-measurement-test PRIVATE simplemc-test)

add_executable(mc-update-test update_test.cpp)
target_link_libraries(mc-update-test PRIVATE simplemc-test)

add_executable(mc-simulation-test simulation_test.cpp)
target_link_libraries(mc-simulation-test PRIVATE simplemc-test)

# add tests
add_test(NAME mc-stats-and-params-test COMMAND mc-stats-and-params-test)
add_test(NAME mc-measurement-test COMMAND mc-measurement-test)
add_test(NAME mc-update-test COMMAND mc-update-test)
add_test(NAME mc-simulation-test COMMAND mc-simulation-test)
