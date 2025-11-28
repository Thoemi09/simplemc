/**
 * @file
 * @brief Implementation details for simplemc/mpi/communicator.hpp.
 */

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/group.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

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

communicator communicator::duplicate() const {
    communicator c;
    check_mpi_call(MPI_Comm_dup(comm_, &c.comm_), "MPI_Comm_dup");
    return c;
}

void communicator::free() {
    check_mpi_call(MPI_Comm_free(&comm_), "MPI_Comm_free");
}

group communicator::get_group() const {
    MPI_Group grp {};
    check_mpi_call(MPI_Comm_group(comm_, &grp), "MPI_Comm_group");
    return group { grp };
}

} // namespace simplemc::mpi
