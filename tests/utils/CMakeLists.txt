# Build tests for simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# utils-test: Executable to test the simplemc-utils library.

# create executable
add_executable(utils-test utils_test.cpp)
target_link_libraries(utils-test PRIVATE simplemc-test)

# add test
add_test(NAME utils-test COMMAND utils-test)
