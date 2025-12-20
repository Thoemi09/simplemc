#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // create a vector and initialize it on all processes
    auto data = std::vector<int>{ 1, 2, 3, 4, 5 };

    // reduce the data in place on root by summing across all processes
    simplemc::mpi::reduce_in_place(data, MPI_SUM, root, comm);

    // print the data on all processes
    fmt::println("Rank {}: {}", comm.rank(), data);
}
