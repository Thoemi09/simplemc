# Build tests for simplemc-serialize-variant.
#
# The following targets are defined:
# ---------------------------------
# variant-serializer-test: Executable to test the backend-erasing variant serializer.

# create executables
add_executable(variant-serializer-test variant_serializer_test.cpp)
target_link_libraries(variant-serializer-test PRIVATE simplemc-test)

# add tests
add_test(NAME variant-serializer-test COMMAND variant-serializer-test)
