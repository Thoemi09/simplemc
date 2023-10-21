# Build mpi library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-mpi (simplemc::simplemc-mpi)

add_library(simplemc-mpi)
add_library(simplemc::simplemc-mpi ALIAS simplemc-mpi)
target_sources(simplemc-mpi
    PRIVATE
        utils.cpp
        environment.cpp
        communicator.cpp)
target_link_libraries(simplemc-mpi 
    PUBLIC 
        simplemc::simplemc-utils
        range-v3::range-v3
        MPI::MPI_CXX)
