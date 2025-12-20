# Build example codes that use simplemc::simplemc-mpi.
#
# The following targets are defined:
# ---------------------------------
# doc_mpi_hello_world: Doc example showing a simple "Hello world" MPI program.
# doc_coll_comm_bcast: Doc example demonstrating the use of broadcast.
# doc_coll_comm_other: Doc example demonstrating the use of all_equal.

add_executable(doc_mpi_hello_world doc_mpi_hello_world.cpp)
target_link_libraries(doc_mpi_hello_world PRIVATE simplemc::simplemc)

add_executable(doc_coll_comm_bcast doc_coll_comm_bcast.cpp)
target_link_libraries(doc_coll_comm_bcast PRIVATE simplemc::simplemc)

add_executable(doc_coll_comm_other doc_coll_comm_other.cpp)
target_link_libraries(doc_coll_comm_other PRIVATE simplemc::simplemc)
