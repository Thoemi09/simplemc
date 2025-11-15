# Build tests for simplemc-grids.
#
# The following targets are defined:
# ---------------------------------
# grids-1d-test: Executable to test the 1-dimensional grids of the simplemc-grids library.
# grids-nd-test: Executable to test the N-dimensional grids of the simplemc-grids library.
# grids-iterator-test: Executable to test the grid_iterator for 1- and N-dimensional grids.

# create executables
add_executable(grids-1d-test grids_1d_test.cpp)
target_link_libraries(grids-1d-test PRIVATE simplemc-test)

add_executable(grids-nd-test grids_nd_test.cpp)
target_link_libraries(grids-nd-test PRIVATE simplemc-test)

add_executable(grids-iterator-test grids_iterator_test.cpp)
target_link_libraries(grids-iterator-test PRIVATE simplemc-test)

# add tests
add_test(NAME grids-1d-test COMMAND grids-1d-test)
add_test(NAME grids-nd-test COMMAND grids-nd-test)
add_test(NAME grids-iterator-test COMMAND grids-iterator-test)
