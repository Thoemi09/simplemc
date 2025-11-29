#include <fmt/base.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

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

// Test duplicating a communicator.
TEST(SimplemcMPI, DuplicateCommunicator) {
    simplemc::mpi::communicator comm;
    simplemc::mpi::communicator dup_comm = comm.duplicate();
    fmt::print("Original communicator: {} of {}\n", comm.rank(), comm.size());
    fmt::print("Duplicated communicator: {} of {}\n", dup_comm.rank(), dup_comm.size());
    int cmp_res {};
    MPI_Comm_compare(comm, dup_comm, &cmp_res);
    ASSERT_NE(MPI_IDENT, cmp_res);
    // No need to call free() - automatic cleanup via shared_ptr
}

// Test thread support query.
TEST(SimplemcMPI, ThreadSupport) {
    int provided = simplemc::mpi::environment::thread_support();
    ASSERT_EQ(provided, MPI_THREAD_SINGLE);
}

// Test is_main_thread function.
TEST(SimplemcMPI, IsMainThread) {
    // in this test, we're in the main thread, so this should return true
    bool is_main = simplemc::mpi::environment::is_main_thread();
    ASSERT_TRUE(is_main);
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
