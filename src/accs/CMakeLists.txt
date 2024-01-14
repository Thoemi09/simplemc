# Build accs library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-accs (simplemc::simplemc-accs)

add_library(simplemc-accs)
add_library(simplemc::simplemc-accs ALIAS simplemc-accs)
target_sources(simplemc-accs
    PRIVATE
        utils.cpp
        mean_acc.cpp)
target_link_libraries(simplemc-accs
    PUBLIC
        simplemc::simplemc-numeric
        simplemc::simplemc-mpi)
