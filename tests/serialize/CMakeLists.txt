# Build tests for simplemc-serialize.
#
# The following targets are defined:
# ---------------------------------
# serializer-test: Exercises the json_serializer class, concept conformance, and ADL-based
# simplemc_save / simplemc_load dispatch.

add_executable(serializer-test serializer_test.cpp)
target_link_libraries(serializer-test PRIVATE simplemc-test)

add_test(NAME serializer-test COMMAND serializer-test)

add_subdirectory(json)
