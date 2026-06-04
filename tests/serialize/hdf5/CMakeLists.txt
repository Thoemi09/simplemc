# Build tests for simplemc-serialize-hdf5.
#
# The following targets are defined:
# ---------------------------------
# hdf5-serializer-test: Executable to test the HDF5 serializer and deserializer.

# create executables
add_executable(hdf5-serializer-test hdf5_serializer_test.cpp)
target_link_libraries(hdf5-serializer-test PRIVATE simplemc-test)

# add tests
add_test(NAME hdf5-serializer-test COMMAND hdf5-serializer-test)
