// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // print the number of processes
    if (comm.rank() == root) {
        fmt::println("Number of processes: {}", comm.size());
    }

    // get the sum of all ranks using all_reduce
    int sum_of_ranks = 0;
    simplemc::mpi::all_reduce(comm.rank(), sum_of_ranks, MPI_SUM, comm);
    fmt::println("Rank {}: Sum of ranks: {}", comm.rank(), sum_of_ranks);

    // get the max. rank using all_reduce_in_place
    int max_rank = comm.rank();
    simplemc::mpi::all_reduce_in_place(max_rank, MPI_MAX, comm);
    fmt::println("Rank {}: Max. rank: {}", comm.rank(), max_rank);

    // gather a single value from all processes into a vector on root
    std::vector<int> vec(comm.size());
    simplemc::mpi::gather(comm.rank(), vec, root, comm);
    fmt::println("Rank {}: Gathered vector: {}", comm.rank(), vec);

    // broadcast the vector to all other processes
    simplemc::mpi::broadcast(vec, root, comm);
    fmt::println("Rank {}: Broadcasted vector: {}", comm.rank(), vec);

    // reduce the broadcasted vector on root
    std::vector<int> reduced_vec(comm.size());
    simplemc::mpi::reduce(vec, reduced_vec, MPI_PROD, root, comm);
    fmt::println("Rank {}: Reduced vector: {}", comm.rank(), reduced_vec);

    // scatter the reduced vector back to all processes
    int scattered_value = 0;
    simplemc::mpi::scatter(reduced_vec, scattered_value, root, comm);
    fmt::println("Rank {}: Scattered value: {}", comm.rank(), scattered_value);
}
