# Build accs library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-accs (simplemc::simplemc-accs): Various accumulators for collecting and analyzing data.

add_library(simplemc-accs INTERFACE)
add_library(simplemc::simplemc-accs ALIAS simplemc-accs)
target_link_libraries(simplemc-accs
    INTERFACE
        simplemc::simplemc-numeric
        simplemc::simplemc-mpi)
