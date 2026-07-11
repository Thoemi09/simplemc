// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ostream.h>
#include <simplemc/mpi.hpp>
#include <simplemc/utils.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    try {
        simplemc::mpi::environment env { argc, argv };
        simplemc::mpi::communicator comm {};
        const int rank = comm.rank();

        // throw an exception on rank 0
        if (rank == 0) {
            throw simplemc::simplemc_exception { "This is a test exception" };
        }

        // on other ranks, just print a message after a delay
        std::this_thread::sleep_for(std::chrono::seconds { 3 });
        fmt::println("Hello from rank {}", rank);
    } catch (const simplemc::simplemc_exception& e) {
        fmt::println(std::cerr, "Caught exception: {}", e.what());
    }
}
