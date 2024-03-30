# Build tests for simplemc-grids.
#
# The following targets are defined:
# ---------------------------------
# grids-test: Executable to test the simplemc-grids library.

# create executable
add_executable(grids-test grids_test.cpp)
target_link_libraries(grids-test PRIVATE simplemc-test simplemc::simplemc-grids)

# add test
add_test(NAME grids-test COMMAND grids-test)
