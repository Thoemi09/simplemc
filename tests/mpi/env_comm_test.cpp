/**
 * @file env_comm_test.cpp
 * @brief Unit tests for the mpi library.
 */

#include <gtest/gtest.h>

#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/core.h>

TEST(SimplemcMPI, HelloWorldWithMPIEnvironment) {
    simplemc::mpi::communicator comm;
    fmt::print("Hello world, from {} of {} processes.\n", comm.rank(), comm.size());
}

TEST(SimplemcMPI, ThrowException) {
    simplemc::mpi::communicator comm;
    try {
        if (comm.rank() == 0) {
            fmt::print("Throwing mpi_exception on rank 0.\n");
            throw simplemc::simplemc_exception("-10", "ThrowException");
        }
    } catch (const std::exception& e) { fmt::print("{}\n", e.what()); }
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (comm.rank() != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    result = RUN_ALL_TESTS();

    return result;
}
