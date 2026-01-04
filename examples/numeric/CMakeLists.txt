# Build example codes that use simplemc::simplemc-numeric.
#
# The following targets are defined:
# ---------------------------------
# doc_lattices: Doc example demonstrating the bravais_lattice class.

add_executable(doc_lattices doc_lattices.cpp)
target_link_libraries(doc_lattices PRIVATE simplemc::simplemc)
