# Build tests for simplemc-accs.
#
# The following targets are defined:
# ---------------------------------
# accs-test: Executable to test the simplemc-accs library.

# create executable
add_executable(accs-test accs_test.cpp)
target_link_libraries(accs-test PRIVATE simplemc-test simplemc::simplemc-accs)

# add test
add_test(NAME accs-test COMMAND accs-test)
