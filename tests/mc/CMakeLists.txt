# Build tests for simplemc-mc.
#
# The following targets are defined:
# ---------------------------------
# mc-stats-and-params-test: Test simulation_params, simulation_stats, and update_stats.
# mc-measurement-test: Test the type-erased measurement wrapper and the mc_measurement concept.
# mc-update-test: Test the type-erased update wrapper and the mc_update concept.
# mc-simulation-test: Test the simulation run-loop driver.
# mc-simulation-input-config-test: Test the input-config flavor of simulation serialization.

# create executables
add_executable(mc-stats-and-params-test stats_and_params_test.cpp)
target_link_libraries(mc-stats-and-params-test PRIVATE simplemc-test)

add_executable(mc-measurement-test measurement_test.cpp)
target_link_libraries(mc-measurement-test PRIVATE simplemc-test)

add_executable(mc-update-test update_test.cpp)
target_link_libraries(mc-update-test PRIVATE simplemc-test)

add_executable(mc-simulation-test simulation_test.cpp)
target_link_libraries(mc-simulation-test PRIVATE simplemc-test)

add_executable(mc-simulation-input-config-test simulation_input_config_test.cpp)
target_link_libraries(mc-simulation-input-config-test PRIVATE simplemc-test)

add_executable(mc-simulation-mixed-backend-test simulation_mixed_backend_test.cpp)
target_link_libraries(mc-simulation-mixed-backend-test PRIVATE simplemc-test)

# add tests
add_test(NAME mc-stats-and-params-test COMMAND mc-stats-and-params-test)
add_test(NAME mc-measurement-test COMMAND mc-measurement-test)
add_test(NAME mc-update-test COMMAND mc-update-test)
add_test(NAME mc-simulation-test COMMAND mc-simulation-test)
add_test(NAME mc-simulation-input-config-test COMMAND mc-simulation-input-config-test)
add_test(NAME mc-simulation-mixed-backend-test COMMAND mc-simulation-mixed-backend-test)
