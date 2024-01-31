/**
 * @file communicator.cpp
 * @brief Wrapper for an MPI_Comm MPI communicator.
 */

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/utils.hpp>

namespace simplemc::mpi {

communicator::communicator(MPI_Comm comm) : comm_(comm) {}

int communicator::rank() const {
    int my_rank {};
    check_mpi_call(MPI_Comm_rank(comm_, &my_rank), "MPI_Comm_rank");
    return my_rank;
}

int communicator::size() const {
    int my_size {};
    check_mpi_call(MPI_Comm_size(comm_, &my_size), "MPI_Comm_size");
    return my_size;
}

void communicator::barrier() const {
    check_mpi_call(MPI_Barrier(comm_), "MPI_Barrier");
}

} // namespace simplemc::mpi
