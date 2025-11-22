# Build example codes that use simplemc::simplemc-grids.
#
# The following targets are defined:
# ---------------------------------
# doc_print_grids: Doc example printing a 1D and 2D grid.
# doc_custom_grid: Doc example demonstrating custom_grid.
# doc_linear_grid: Doc example demonstrating linear_grid.
# doc_power_grid: Doc example demonstrating power_grid.
# doc_symmetric_power_grid: Doc example demonstrating symmetric_power_grid.

add_executable(doc_print_grids doc_print_grids.cpp)
target_link_libraries(doc_print_grids PRIVATE simplemc::simplemc)

add_executable(doc_custom_grid doc_custom_grid.cpp)
target_link_libraries(doc_custom_grid PRIVATE simplemc::simplemc)

add_executable(doc_linear_grid doc_linear_grid.cpp)
target_link_libraries(doc_linear_grid PRIVATE simplemc::simplemc)

add_executable(doc_power_grid doc_power_grid.cpp)
target_link_libraries(doc_power_grid PRIVATE simplemc::simplemc)

add_executable(doc_symmetric_power_grid doc_symmetric_power_grid.cpp)
target_link_libraries(doc_symmetric_power_grid PRIVATE simplemc::simplemc)
