#include <fmt/ostream.h>
#include <simplemc/mpi.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    try {
        simplemc::mpi::environment env(argc, argv);
        simplemc::mpi::communicator comm;
        int rank = comm.rank();
        if (rank == 0) {
            throw simplemc::simplemc_exception("This is a test exception");
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
        fmt::print("Hello from rank {}\n", rank);
    } catch(const simplemc::simplemc_exception& e) {
        fmt::print(std::cerr, "Caught exception: {}\n", e.what());
    }
}
