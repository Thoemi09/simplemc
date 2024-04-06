# Build tests for simplemc-serialization.
#
# The following targets are defined:
# ---------------------------------
# serializtion-test: Executable to test the simplemc-serialization library.

# create executable
add_executable(serialization-test serialization_test.cpp)
target_link_libraries(serialization-test PRIVATE simplemc-test simplemc::simplemc-serialization)

# add test
add_test(NAME serialization-test COMMAND serialization-test)
