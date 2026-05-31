# Build tests for simplemc-serialize-json.
#
# The following targets are defined:
# ---------------------------------
# serializers-test: Per-library-type round-trip tests through the JSON backend.

add_executable(serializers-test serializers_test.cpp)
target_link_libraries(serializers-test PRIVATE simplemc-test)

add_test(NAME serializers-test COMMAND serializers-test)
