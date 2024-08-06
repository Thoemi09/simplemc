#include <gtest/gtest.h>

#include <simplemc/mpi.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/core.h>

#include <exception>

// Test rank and size of communicator.
TEST(SimplemcMPI, HelloWorldWithMPIEnvironment) {
    simplemc::mpi::communicator comm;
    fmt::print("Hello world, from {} of {} processes.\n", comm.rank(), comm.size());
}

// Test throwing and catching an exception on some process.
TEST(SimplemcMPI, ThrowException) {
    simplemc::mpi::communicator comm;
    try {
        if (comm.rank() == 0) {
            fmt::print("Throwing a simplemc::simplemc_exception on rank 0.\n");
            throw simplemc::simplemc_exception("-10", "ThrowException");
        }
    } catch (const std::exception& e) {
        fmt::print("{}\n", e.what());
    }
}

// Custom main function for MPI.
int main(int argc, char* argv[]) {
    int result = 0;

    // Initialize GoogleTest and MPI.
    ::testing::InitGoogleTest(&argc, argv);
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    // Remove the default GoogleTest listener on all ranks except 0.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm.rank() != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // Run the tests.
    result = RUN_ALL_TESTS();

    return result;
}
