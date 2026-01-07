# Build example codes that use simplemc::simplemc-grids.
#
# The following targets are defined:
# ---------------------------------
# doc_print_grids: Doc example printing a 1D and 2D grid.
# doc_custom_grid: Doc example demonstrating custom_grid.
# doc_linear_grid: Doc example demonstrating linear_grid.
# doc_power_grid: Doc example demonstrating power_grid.
# doc_symmetric_power_grid: Doc example demonstrating symmetric_power_grid.
# doc_nd_grid: Doc example demonstrating nd_grid.

simplemc_add_examples(
    doc_print_grids
    doc_custom_grid
    doc_linear_grid
    doc_power_grid
    doc_symmetric_power_grid
    doc_nd_grid
)
