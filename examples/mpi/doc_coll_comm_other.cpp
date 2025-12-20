#include <fmt/base.h>
#include <simplemc/mpi.hpp>

#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};    

    // initialize a vector on all processes with the same data
    auto data = std::vector<int>{1, 2, 3, 4, 5};

    // check that the data is equal on all processes
    bool equal = simplemc::mpi::all_equal(data, comm);
    if (comm.rank() == 0) {
        fmt::println("Initial data is equal? --> {}", equal);
    }

    // change the data to be rank specific
    data[0] = comm.rank();

    // check that the data is not equal on all processes
    equal = simplemc::mpi::all_equal(data, comm);
    if (comm.rank() == 0) {
        fmt::println("Modified data is equal? --> {}", equal);
    }
}
