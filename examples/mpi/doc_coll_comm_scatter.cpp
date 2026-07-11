// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <cstddef>
#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // create the send buffer on root (3 elements per process)
    const int count = 3;
    const int total_count = count * comm.size();
    std::vector<int> send_data {};
    if (comm.rank() == root) {
        for (int i = 0; i < total_count; ++i) {
            send_data.push_back(i + 1);
        }
    }

    // create a receive buffer on all processes
    std::vector<int> recv_data(count);

    // scatter the data from root to all processes
    simplemc::mpi::scatter(send_data, recv_data, root, comm);

    // print the received data on all processes
    fmt::println("Rank {}: {}", comm.rank(), recv_data);
}
