// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/mpi.hpp>

int main(int argc, char** argv) {
    // initialize the MPI environment and the communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};

    // print the number of processes on process 0
    if (comm.rank() == 0) {
        fmt::println("Number of processes: {}", comm.size());
    }

    // print message on each process
    fmt::println("Hello world, from process {} out of {} processes.", comm.rank(), comm.size());
}
