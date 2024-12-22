# Build tests for simplemc-json.
#
# The following targets are defined:
# ---------------------------------
# json-test: Executable to the simplemc-json library.

# create executable
add_executable(json-test json_test.cpp)
target_link_libraries(json-test PRIVATE simplemc-test)

# add test
add_test(NAME json-test COMMAND json-test)
