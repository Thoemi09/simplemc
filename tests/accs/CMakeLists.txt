# Build tests for simplemc-accs.
#
# The following targets are defined:
# ---------------------------------
# accs-utils-test: Executable to test the utilities of the simplemc-accs library.

add_executable(accs-utils-test accs_utils_test.cpp)
target_link_libraries(accs-utils-test PRIVATE simplemc-test simplemc::simplemc-accs)
add_test(NAME accs-utils-test COMMAND accs-utils-test)

# add_executable(accs-test accs_test.cpp)
# target_link_libraries(accs-test PRIVATE simplemc-test simplemc::simplemc-accs)
# add_test(NAME accs-test COMMAND accs-test)
