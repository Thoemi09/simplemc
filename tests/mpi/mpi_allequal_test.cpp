#include "../gtest-mpi-listener.hpp"

#include <gtest/gtest.h>
#include <simplemc/mpi.hpp>

#include <complex>
#include <vector>

// Test all_equal for scalars.
TEST(SimplemcMPI, AllEqualScalars) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();

    ASSERT_TRUE(simplemc::mpi::all_equal(1000, comm));
    ASSERT_EQ(simplemc::mpi::all_equal(rank, comm), size == 1);
    ASSERT_TRUE(simplemc::mpi::all_equal(3.14159, comm));
    ASSERT_EQ(simplemc::mpi::all_equal(3.14159 + rank, comm), size == 1);
    ASSERT_TRUE(simplemc::mpi::all_equal(std::complex<double>(1.0, 2.0), comm));
    ASSERT_EQ(simplemc::mpi::all_equal(std::complex<double>(rank, 2.0), comm), size == 1);
}

// Test all_equal for ranges.
TEST(SimplemcMPI, AllEqualRanges) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();

    // empty range
    ASSERT_TRUE(simplemc::mpi::all_equal(std::vector<char>(), comm));

    // equal ranges on all ranks
    auto data = std::vector<double> { 1.0, 2.0, 3.0, 4.0 };
    ASSERT_TRUE(simplemc::mpi::all_equal(data, comm));

    // range is different on one rank
    data[0] = rank;
    ASSERT_EQ(simplemc::mpi::all_equal(data, comm), size == 1);

    // ranges of different sizes
    auto data_2 = std::vector<int>(rank + 1, 42);
    ASSERT_EQ(simplemc::mpi::all_equal(data_2, comm), size == 1);
}

// Custom main function for MPI.
int main(int argc, char** argv) {
    // filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // initialize MPI
    MPI_Init(&argc, &argv);

    // add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    // run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
    // pass and 1 if some test fails.
    int result = RUN_ALL_TESTS();

    return result;
}
