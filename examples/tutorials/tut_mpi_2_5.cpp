// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/mpi.hpp>
#include <simplemc/utils.hpp>

int main(int argc, char** argv) {
    try {
        // initialize the MPI environment and the communicator
        simplemc::mpi::environment env { argc, argv };
        simplemc::mpi::communicator comm {};
        const int root = -1;

        // set the error handler to return errors instead of aborting
        simplemc::mpi::comm_set_errhandler(comm, MPI_ERRORS_RETURN);

        // broadcast a message
        int msg = (comm.rank() == 0) ? 42 : 0;
        simplemc::mpi::broadcast(msg, root, comm);

        // print the message on each process
        fmt::println("Process {} received message: {}", comm.rank(), msg);
    } catch (const simplemc::simplemc_exception& e) {
        fmt::println("Caught exception: {}", e.what());
    }
}
