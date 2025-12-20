#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    constexpr int root = 0;

    // create a vector and initialize it on root
    std::vector<int> data(5);
    if (comm.rank() == root) {
        data = { 1, 2, 3, 4, 5 };
    }

    // broadcast the data from root to all other processes
    simplemc::mpi::broadcast(data, root, comm);

    // print the data on all processes
    fmt::println("Rank {}: {}", comm.rank(), data);
}
