# Build tests for simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# utils-general-test: Executable to test the general utilities in the simplemc-utils library.
# utils-exceptions-test: Executable to test the generic exceptions in the simplemc-utils library.
# utils-timer-test: Executable to test the timer utilities in the simplemc-utils library.
# utils-nd-indexing-test: Executable to test the indexing utilities in the simplemc-utils library.

# create executables
add_executable(utils-general-test utils_general_test.cpp)
target_link_libraries(utils-general-test PRIVATE simplemc-test)

add_executable(utils-exceptions-test utils_exceptions_test.cpp)
target_link_libraries(utils-exceptions-test PRIVATE simplemc-test)

add_executable(utils-timer-test utils_timer_test.cpp)
target_link_libraries(utils-timer-test PRIVATE simplemc-test)

add_executable(utils-nd-indexing-test utils_nd_indexing_test.cpp)
target_link_libraries(utils-nd-indexing-test PRIVATE simplemc-test)

# add tests
add_test(NAME utils-general-test COMMAND utils-general-test)
add_test(NAME utils-exceptions-test COMMAND utils-exceptions-test)
add_test(NAME utils-timer-test COMMAND utils-timer-test)
add_test(NAME utils-nd-indexing-test COMMAND utils-nd-indexing-test)
