# Build tests for simplemc-random.
#
# The following targets are defined:
# ---------------------------------
# random-rngs-test: Executable to test the RNGs of the simplemc-random library.
# random-samples-test: Executable to test the random samples of the simplemc-random library.

# create executables
add_executable(random-rngs-test random_rngs_test.cpp)
target_link_libraries(random-rngs-test PRIVATE simplemc-test)

add_executable(random-samples-test random_samples_test.cpp)
target_link_libraries(random-samples-test PRIVATE simplemc-test)

# add tests
add_test(NAME random-rngs-test COMMAND random-rngs-test)
add_test(NAME random-samples-test COMMAND random-samples-test)
