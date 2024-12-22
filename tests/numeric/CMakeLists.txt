# Build tests for simplemc-numeric.
#
# The following targets are defined:
# ---------------------------------
# numeric-utils-test: Executable to test the utilities of the simplemc-numeric.
# numeric-1d-lattices-test: Executable to test the 1D lattices of the simplemc-numeric.
# numeric-2d-lattices-test: Executable to test the 2D lattices of the simplemc-numeric.
# numeric-3d-lattices-test: Executable to test the 3D lattices of the simplemc-numeric.
# numeric-interpolation-test: Executable to test the interpolation tools of simplemc-numeric.

add_executable(numeric-utils-test numeric_utils_test.cpp)
target_link_libraries(numeric-utils-test PRIVATE simplemc-test)
add_test(NAME numeric-utils-test COMMAND numeric-utils-test)

add_executable(numeric-1d-lattices-test numeric_1d_lattices_test.cpp)
target_link_libraries(numeric-1d-lattices-test PRIVATE simplemc-test)
add_test(NAME numeric-1d-lattices-test COMMAND numeric-1d-lattices-test)

add_executable(numeric-2d-lattices-test numeric_2d_lattices_test.cpp)
target_link_libraries(numeric-2d-lattices-test PRIVATE simplemc-test)
add_test(NAME numeric-2d-lattices-test COMMAND numeric-2d-lattices-test)

add_executable(numeric-3d-lattices-test numeric_3d_lattices_test.cpp)
target_link_libraries(numeric-3d-lattices-test PRIVATE simplemc-test)
add_test(NAME numeric-3d-lattices-test COMMAND numeric-3d-lattices-test)

add_executable(numeric-interpolation-test numeric_interpolation_test.cpp)
target_link_libraries(numeric-interpolation-test PRIVATE simplemc-test)
add_test(NAME numeric-interpolation-test COMMAND numeric-interpolation-test)

add_executable(numeric-quadrature-test numeric_quadrature_test.cpp)
target_link_libraries(numeric-quadrature-test PRIVATE simplemc-test)
add_test(NAME numeric-quadrature-test COMMAND numeric-quadrature-test)

add_executable(numeric-special-functions-test numeric_special_functions_test.cpp)
target_link_libraries(numeric-special-functions-test PRIVATE simplemc-test)
add_test(NAME numeric-special-functions-test COMMAND numeric-special-functions-test)
