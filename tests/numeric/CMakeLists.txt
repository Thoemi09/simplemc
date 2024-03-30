# Build tests for simplemc-numeric.
#
# The following targets are defined:
# ---------------------------------
# numeric-test: Executable to test the simplemc-numeric library.

# create executable
add_executable(numeric-test numeric_test.cpp)
target_link_libraries(numeric-test PRIVATE simplemc-test simplemc::simplemc-numeric)

# add test
add_test(NAME numeric-test COMMAND numeric-test)
