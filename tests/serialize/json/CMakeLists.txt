# Build tests for simplemc-serialize-json.
#
# The following targets are defined:
# ---------------------------------
# json-serializer-test: Executable to test the JSON serializer and deserializer.
# json-utils-test: Executable to test utility functions for JSON serialization.

# create executables
add_executable(json-serializer-test json_serializer_test.cpp)
target_link_libraries(json-serializer-test PRIVATE simplemc-test)

add_executable(json-utils-test json_utils_test.cpp)
target_link_libraries(json-utils-test PRIVATE simplemc-test)

# add tests
add_test(NAME json-serializer-test COMMAND json-serializer-test)
add_test(NAME json-utils-test COMMAND json-utils-test)
