# Build example codes that use simplemc::simplemc-grids.
#
# The following targets are defined:
# ---------------------------------
# doc_print_grids: Doc example printing a 1D and 2D grid.

add_executable(doc_print_grids doc_print_grids.cpp)
target_link_libraries(doc_print_grids PRIVATE simplemc::simplemc)