/**
 * @file
 * @brief Implementation details for simplemc/mpi/communicator.hpp.
 */

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/group.hpp>

#include <mpi.h>

namespace simplemc::mpi {

communicator::communicator() : comm_(std::make_shared<MPI_Comm>(MPI_COMM_WORLD)) {}

communicator::communicator(MPI_Comm comm, resource_policy pol) {
    if (pol == resource_policy::take_ownership) {
        comm_ = std::shared_ptr<MPI_Comm> { new MPI_Comm(comm), comm_deleter {} };
    } else {
        comm_ = std::make_shared<MPI_Comm>(comm);
    }
}

communicator communicator::duplicate(resource_policy pol) const {
    MPI_Comm new_comm {};
    check_mpi_call(MPI_Comm_dup(*comm_, &new_comm), "MPI_Comm_dup");
    return communicator { new_comm, pol };
}

communicator communicator::create(const group& grp, resource_policy pol) const {
    MPI_Comm new_comm {};
    check_mpi_call(MPI_Comm_create(*comm_, grp, &new_comm), "MPI_Comm_create");
    return communicator { new_comm, pol };
}

int communicator::rank() const {
    int my_rank {};
    check_mpi_call(MPI_Comm_rank(*comm_, &my_rank), "MPI_Comm_rank");
    return my_rank;
}

int communicator::size() const {
    int my_size {};
    check_mpi_call(MPI_Comm_size(*comm_, &my_size), "MPI_Comm_size");
    return my_size;
}

group communicator::get_group(resource_policy pol) const {
    MPI_Group grp {};
    check_mpi_call(MPI_Comm_group(*comm_, &grp), "MPI_Comm_group");
    return group { grp, pol };
}

void communicator::barrier() const {
    check_mpi_call(MPI_Barrier(*comm_), "MPI_Barrier");
}

communicator communicator::split(int color, int key, resource_policy pol) const {
    MPI_Comm new_comm {};
    check_mpi_call(MPI_Comm_split(*comm_, color, key, &new_comm), "MPI_Comm_split");
    return communicator { new_comm, pol };
}

void communicator::comm_deleter::operator()(MPI_Comm* c) const {
    // Only free if valid and not a predefined communicator.
    if (!environment::is_finalized() && c && *c != MPI_COMM_NULL && *c != MPI_COMM_WORLD && *c != MPI_COMM_SELF) {
        MPI_Comm_free(c);
    }
    delete c;
}

void comm_free(MPI_Comm& comm) {
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF) {
        check_mpi_call(MPI_Comm_free(&comm), "MPI_Comm_free");
    }
}

int comm_compare(const communicator& c1, const communicator& c2) {
    int result {};
    check_mpi_call(MPI_Comm_compare(c1, c2, &result), "MPI_Comm_compare");
    return result;
}

} // namespace simplemc::mpi
