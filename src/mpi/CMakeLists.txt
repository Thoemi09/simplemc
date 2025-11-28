# Build mpi library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-mpi (simplemc::simplemc-mpi): C++ wrapper around MPI C implementations.

add_library(simplemc-mpi)
add_library(simplemc::simplemc-mpi ALIAS simplemc-mpi)
target_sources(simplemc-mpi
    PRIVATE
        communicator.cpp
        environment.cpp
        group.cpp
        utils.cpp)
target_link_libraries(simplemc-mpi
    PUBLIC
        simplemc::simplemc-utils
        MPI::MPI_CXX)
