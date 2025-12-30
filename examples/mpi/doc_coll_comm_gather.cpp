#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // create a vector with rank-dependent size (each process has rank + 1 elements)
    std::vector<int> send_data(comm.rank() + 1);
    for (int i = 0; i < comm.rank() + 1; ++i) {
        send_data[i] = comm.rank() * 10 + i;
    }

    // create the receive buffer on root (only root needs it)
    // total size is 1 + 2 + 3 + ... + comm.size() = comm.size() * (comm.size() + 1) / 2
    std::vector<int> recv_data {};
    if (comm.rank() == root) {
        recv_data.resize(comm.size() * (comm.size() + 1) / 2);
    }

    // gather all data on root
    simplemc::mpi::gatherv(send_data, recv_data, root, comm);

    // print the result on root
    if (comm.rank() == root) {
        fmt::println("Gathered data: {}", recv_data);
    }
}
