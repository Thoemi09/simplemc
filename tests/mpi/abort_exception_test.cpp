/**
 * @file abort_exception_test.cpp
 * @brief Test to abort MPI program on exception.
 */

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
        std::cout << "Hello from rank " << rank << std::endl;
    } catch(const simplemc::simplemc_exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}
