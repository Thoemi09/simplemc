# Build tests for simplemc-random.
#
# The following targets are defined:
# ---------------------------------
# random-test: Executable to test the simplemc-random library.

# create executable
add_executable(random-test random_test.cpp)
target_link_libraries(random-test PRIVATE simplemc-test simplemc::simplemc-random)

# add test
add_test(NAME random-test COMMAND random-test)
