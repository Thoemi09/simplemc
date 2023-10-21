#include <simplemc/mpi.hpp>
#include <iostream>

int main(int argc, char** argv) {
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    int rank = comm.rank();
    int res = 0;
    simplemc::mpi::all_reduce(comm, rank, res, MPI_SUM);

    if (rank == 0) {
        std::cout << "Result: " << res << std::endl;
    }
}
