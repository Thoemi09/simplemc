// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <array>
#include <complex>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

// Test broadcast for a single value of type T.
template <typename T>
void check_single_broadcast(const T& exp) {
    using namespace simplemc::mpi;
    communicator comm {};
    const int rank = comm.rank();

    // broadcast overloads to test
    auto fvec = std::vector<std::function<void(T&, int)>> {};
    fvec.push_back([&](T& value, int root) { broadcast(&value, 1, mpi_type<T>::get(), root, comm); });
    fvec.push_back([&](T& value, int root) { broadcast(&value, 1, root, comm); });
    fvec.push_back([&](T& value, int root) { broadcast(value, root, comm); });

    // perform broadcast and check the result
    for (int r = 0; r < comm_size(comm); ++r) {
        for (auto fun : fvec) {
            T value = (rank == r) ? exp : T {};
            fun(value, r);
            ASSERT_EQ(value, exp);
        }
    }
}

// Test broadcast for a range of type R.
template <typename R>
void check_range_broadcast(const R& exp) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    const int rank = comm.rank();
    const auto count = simplemc::ranges::size(exp);
    auto mpi_dtype = mpi_type<value_type>::get();

    // broadcast overloads to test
    auto fvec = std::vector<std::function<void(R&, int)>> {};
    fvec.push_back([&](R& rg, int root) { broadcast(simplemc::ranges::data(rg), count, mpi_dtype, root, comm); });
    fvec.push_back([&](R& rg, int root) { broadcast(simplemc::ranges::data(rg), count, root, comm); });
    fvec.push_back([&](R& rg, int root) { broadcast(rg, root, comm); });

    // perform broadcast and check the result
    for (int r = 0; r < comm.size(); ++r) {
        for (auto fun : fvec) {
            R data {};
            if (rank == r) {
                data = exp;
            } else {
                data.resize(exp.size());
            }
            fun(data, r);
            ASSERT_EQ(data, exp);
        }
    }
}

TEST(SimplemcMPI, BroadcastZeroValues) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int root = 0;
    int value = rank;
    simplemc::mpi::broadcast(&value, 0, MPI_INT, root, comm);
    ASSERT_EQ(value, rank);
}

TEST(SimplemcMPI, BroadCastSingleValues) {
    using std::numbers::pi;

    // byte type
    check_single_broadcast<std::byte>(std::byte { 0x7F });

    // character types
    check_single_broadcast<char>('x');
    check_single_broadcast<signed char>('a');
    check_single_broadcast<unsigned char>('B');

    // signed integer types
    check_single_broadcast<short>(-100);
    check_single_broadcast<int>(-12345);
    check_single_broadcast<long>(-123456L);
    check_single_broadcast<long long>(-12345678LL);

    // unsigned integer types
    check_single_broadcast<unsigned short>(100);
    check_single_broadcast<unsigned int>(12345U);
    check_single_broadcast<unsigned long>(123456UL);
    check_single_broadcast<unsigned long long>(12345678ULL);

    // floating point types
    check_single_broadcast<float>(static_cast<float>(pi));
    check_single_broadcast<double>(-pi);
    check_single_broadcast<long double>(pi * pi);

    // complex types
    check_single_broadcast<std::complex<float>>({ 1.0f, -2.0f });
    check_single_broadcast<std::complex<double>>({ 1.0, 2.0 });
    check_single_broadcast<std::complex<long double>>({ -pi, pi });
}

TEST(SimplemcMPI, BroadcastEmptyVector) {
    simplemc::mpi::communicator comm {};
    const int root = 0;
    std::vector<int> data {};
    simplemc::mpi::broadcast(data, root, comm);
    ASSERT_TRUE(data.empty());
}

TEST(SimplemcMPI, BroadcastRanges) {
    using std::numbers::pi;

    // byte type
    check_range_broadcast(std::vector<std::byte> { std::byte { 0x00 }, std::byte { 0x7F }, std::byte { 0xFF } });

    // character types
    check_range_broadcast(std::vector<char> { 'a' });
    check_range_broadcast(std::vector<signed char> { 'a', 'b' });
    check_range_broadcast(std::vector<unsigned char> { 'a', 'b', 'c' });

    // signed integer types
    check_range_broadcast(std::vector<short> { -1, 1, 20 });
    check_range_broadcast(std::vector<unsigned short> { 1, 20 });
    check_range_broadcast(std::vector<int> { -1 });

    // unsigned integer types
    check_range_broadcast(std::vector<unsigned int> { 1, 20 });
    check_range_broadcast(std::vector<long> { -1, 1, 20 });
    check_range_broadcast(std::vector<unsigned long> { 1, 20 });
    check_range_broadcast(std::vector<long long> { -1 });
    check_range_broadcast(std::vector<unsigned long long> { 200, 1, 20 });

    // floating point types
    check_range_broadcast(std::vector<float> { -pi });
    check_range_broadcast(std::vector<double> { pi, -2 * pi });
    check_range_broadcast(std::vector<long double> { 2 * pi, -pi, pi * pi });

    // complex types
    check_range_broadcast(std::vector<std::complex<float>> { { 0.0f, 1.0f }, { -2.0f, 3.0f }, { 4.0f, 0.0f } });
    check_range_broadcast(std::vector<std::complex<double>> { { 1.0, 1.0 }, { -2.0, 3.0 } });
    check_range_broadcast(std::vector<std::complex<long double>> { { pi, -pi } });

    // string
    check_range_broadcast(std::string { "Hello world!" });
}

TEST(SimplemcMPI, BroadcastLargeArray) {
    simplemc::mpi::communicator comm {};
    const int root = 0;
    constexpr int count = 10000;
    auto iota = simplemc::ranges::iota_view(0, count);
    auto arr = std::array<int, count> {};
    if (comm.rank() == root) {
        simplemc::ranges::copy(iota, arr.begin());
    }
    simplemc::mpi::broadcast(arr, root, comm);
    check_equal(arr, iota);
}

TEST(SimplemcMPI, BroadcastWithSplitCommunicator) {
    simplemc::mpi::communicator comm {};
    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // split into even and odd ranks
    const int color = comm.rank() % 2;
    auto sub_comm = comm.split(color);

    // broadcast different values in each sub-communicator
    const int root = 0;
    int value = (sub_comm.rank() == root) ? (color == 0 ? 100 : 200) : 0;

    // broadcast within sub-communicator and check result
    simplemc::mpi::broadcast(value, root, sub_comm);
    ASSERT_EQ(value, (color == 0 ? 100 : 200));
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
