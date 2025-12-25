#include <simplemc/mpi.hpp>
#include <simplemc/utils.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    try {
        simplemc::mpi::environment env(argc, argv, false);
        simplemc::mpi::communicator comm;
        int rank = comm.rank();
        if (rank != 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
        } else {
            throw simplemc::simplemc_exception("This is a test exception");
        }
    } catch(const simplemc::simplemc_exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}
