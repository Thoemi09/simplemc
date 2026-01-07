# Build example codes that use simplemc::simplemc-mpi.
#
# The following targets are defined:
# ---------------------------------
# doc_mpi_hello_world: Doc example showing a simple "Hello world" MPI program.
# doc_coll_comm_bcast: Doc example demonstrating the use of broadcast.
# doc_coll_comm_gather: Doc example demonstrating the use of gatherv.
# doc_coll_comm_other: Doc example demonstrating the use of all_equal.
# doc_coll_comm_reduce: Doc example demonstrating the use of reduce.
# doc_coll_comm_scatter: Doc example demonstrating the use of scatter.

simplemc_add_examples(
    doc_mpi_hello_world
    doc_coll_comm_bcast
    doc_coll_comm_gather
    doc_coll_comm_other
    doc_coll_comm_reduce
    doc_coll_comm_scatter
)
