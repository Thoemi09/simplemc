# Build example codes that use simplemc::simplemc-mpi.
#
# The following targets are defined:
# ---------------------------------
# doc_mpi_hello_world: Doc example showing a simple "Hello world" MPI program.

add_executable(doc_mpi_hello_world doc_mpi_hello_world.cpp)
target_link_libraries(doc_mpi_hello_world PRIVATE simplemc::simplemc)
