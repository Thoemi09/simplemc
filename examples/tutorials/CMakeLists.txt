# Build tutorial codes that use simplemc::simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# tut_utils_1: Using simplemc::timer to measure runtime.
# tut_utils_2: Multi-dimensional indexing.
# tut_utils_3: Using fmt to print formatted output.
# tut_mpi_1: "Hello, World!" using simplemc-mpi.
# tut_mpi_2_1: Error handling: Working example that broadcasts a simple message.
# tut_mpi_2_2: Error handling: Using an invalid root rank.
# tut_mpi_2_3: Error handling: Setting the MPI error handler to return errors.
# tut_mpi_2_4: Error handling: Catch exceptions locally.
# tut_mpi_2_5: Error handling: Abort on exception set to true.
# tut_mpi_2_6: Error handling: Abort on exception set to false.
# tut_mpi_3: Common collective MPI operations.
# tut_mpi_4: Custom datatypes with simplemc-mpi.
# tut_mpi_5: Groups and communicators with simplemc-mpi.
# tut_random_1: Generating random samples with simplemc-random.
# tut_grids_1: Various 1-dimensional grids.
# tut_grids_2: N-dimensional grids.
# tut_numeric_1: Bravais lattices.
# tut_numeric_2: Using orthogonal polynomials.

simplemc_add_examples(
    tut_utils_1
    tut_utils_2
    tut_utils_3
    tut_mpi_1
    tut_mpi_2_1
    tut_mpi_2_2
    tut_mpi_2_3
    tut_mpi_2_4
    tut_mpi_2_5
    tut_mpi_2_6
    tut_mpi_3
    tut_mpi_4
    tut_mpi_5
    tut_random_1
    tut_grids_1
    tut_grids_2
    tut_numeric_1
    tut_numeric_2
)
