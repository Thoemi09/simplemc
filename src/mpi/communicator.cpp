// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation details for simplemc/mpi/communicator.hpp.
 */

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/group.hpp>
#include <simplemc/mpi/utils.hpp>

#include <memory>
#include <mpi.h>

namespace simplemc::mpi {

void comm_free(MPI_Comm& comm) {
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF) {
        check_mpi_call(MPI_Comm_free(&comm), "MPI_Comm_free");
    }
}

int comm_size(MPI_Comm comm) {
    int sz {};
    check_mpi_call(MPI_Comm_size(comm, &sz), "MPI_Comm_size");
    return sz;
}

int comm_rank(MPI_Comm comm) {
    int rk {};
    check_mpi_call(MPI_Comm_rank(comm, &rk), "MPI_Comm_rank");
    return rk;
}

MPI_Group comm_group(MPI_Comm comm) {
    MPI_Group grp {};
    check_mpi_call(MPI_Comm_group(comm, &grp), "MPI_Comm_group");
    return grp;
}

MPI_Comm comm_dup(MPI_Comm comm) {
    MPI_Comm new_comm {};
    check_mpi_call(MPI_Comm_dup(comm, &new_comm), "MPI_Comm_dup");
    return new_comm;
}

MPI_Comm comm_split(MPI_Comm comm, int color, int key) {
    MPI_Comm new_comm {};
    check_mpi_call(MPI_Comm_split(comm, color, key, &new_comm), "MPI_Comm_split");
    return new_comm;
}

MPI_Comm comm_create(MPI_Comm comm, MPI_Group grp) {
    MPI_Comm new_comm {};
    check_mpi_call(MPI_Comm_create(comm, grp, &new_comm), "MPI_Comm_create");
    return new_comm;
}

int comm_compare(MPI_Comm comm1, MPI_Comm comm2) {
    int result {};
    check_mpi_call(MPI_Comm_compare(comm1, comm2, &result), "MPI_Comm_compare");
    return result;
}

void comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) {
    check_mpi_call(MPI_Comm_set_errhandler(comm, errhandler), "MPI_Comm_set_errhandler");
}

communicator::communicator(MPI_Comm comm, resource_policy pol) {
    if (pol == resource_policy::take_ownership) {
        comm_ = std::shared_ptr<MPI_Comm> { new MPI_Comm(comm), comm_deleter {} };
    } else {
        comm_ = std::make_shared<MPI_Comm>(comm);
    }
}

void communicator::comm_deleter::operator()(MPI_Comm* c) const noexcept {
    if (!finalized() && c && *c != MPI_COMM_NULL && *c != MPI_COMM_WORLD && *c != MPI_COMM_SELF) {
        MPI_Comm_free(c);
    }
    delete c;
}

} // namespace simplemc::mpi
