#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <fmt/format.h>
#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <functional>
#include <vector>

// Test all-gathering a single value of type T.
template <typename T>
void check_single_all_gather(const T& input, const std::vector<T>& expected) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto mpi_dtype = mpi_type<T>::get();
    auto fvec = std::vector<std::function<void(const T&, std::vector<T>&)>> {};
    fvec.push_back(
        [&](const T& in, std::vector<T>& out) { all_gather(&in, 1, mpi_dtype, out.data(), 1, mpi_dtype, comm); });
    fvec.push_back([&](const T& in, std::vector<T>& out) { all_gather(&in, out.data(), 1, comm); });
    fvec.push_back([&](const T& in, std::vector<T>& out) { all_gather(in, out, comm); });
    for (auto all_gather : fvec) {
        std::vector<T> result(comm.size());
        all_gather(input, result);
        ASSERT_EQ(result, expected);
    }
}

// Test all-gathering a single value in place of type T.
template <typename T>
void check_single_all_gather_in_place(const std::vector<T>& input, const std::vector<T>& expected) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto fvec = std::vector<std::function<void(std::vector<T>&)>> {};
    fvec.push_back([&](std::vector<T>& data) { all_gather_in_place(data.data(), 1, mpi_type<T>::get(), comm); });
    fvec.push_back([&](std::vector<T>& data) { all_gather_in_place(data.data(), 1, comm); });
    fvec.push_back([&](std::vector<T>& data) { all_gather_in_place(data, comm); });
    for (auto all_gather_in_place : fvec) {
        std::vector<T> data = input;
        all_gather_in_place(data);
        ASSERT_EQ(data, expected);
    }
}

// Test all-gathering a range of type R.
template <typename R>
void check_range_all_gather(const R& input, const R& expected) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const size = simplemc::ranges::size(input);
    auto mpi_dtype = mpi_type<value_type>::get();
    auto fvec = std::vector<std::function<void(const R&, R&)>> {};
    fvec.push_back([&](const R& in_rg, R& out_rg) {
        all_gather(
            simplemc::ranges::data(in_rg), size, mpi_dtype, simplemc::ranges::data(out_rg), size, mpi_dtype, comm);
    });
    fvec.push_back([&](const R& in_rg, R& out_rg) {
        all_gather(simplemc::ranges::data(in_rg), simplemc::ranges::data(out_rg), size, comm);
    });
    fvec.push_back([&](const R& in_rg, R& out_rg) { all_gather(in_rg, out_rg, comm); });
    for (auto all_gather : fvec) {
        R result {};
        result.resize(simplemc::ranges::size(expected));
        all_gather(input, result);
        ASSERT_EQ(result, expected);
    }
}

// Test all-gathering a range in place of type R.
template <typename R>
void check_range_all_gather_in_place(const R& input, const R& expected) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const total_size = simplemc::ranges::size(input);
    auto const count_per_process = total_size / comm.size(); // Count to receive from each process
    auto mpi_dtype = mpi_type<value_type>::get();
    auto fvec = std::vector<std::function<void(R&)>> {};
    fvec.push_back([&](R& rg) { all_gather_in_place(simplemc::ranges::data(rg), count_per_process, mpi_dtype, comm); });
    fvec.push_back([&](R& rg) { all_gather_in_place(simplemc::ranges::data(rg), count_per_process, comm); });
    fvec.push_back([&](R& rg) { all_gather_in_place(rg, comm); });
    for (auto all_gather_in_place : fvec) {
        R data = input;
        all_gather_in_place(data);
        ASSERT_EQ(data, expected);
    }
}

TEST(SimplemcMPI, AllGatherZeroValues) {
    simplemc::mpi::communicator comm {};
    int in_value = comm.rank();
    std::vector<int> out_values(comm.size(), -1);
    simplemc::mpi::all_gather(&in_value, 0, MPI_INT, out_values.data(), 0, MPI_INT, comm);
    for (int i = 0; i < comm.size(); ++i) {
        ASSERT_EQ(out_values[i], -1);
    }
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single() {
    simplemc::mpi::communicator comm {};
    const T rank = simplemc::mpi::comm_rank(comm);
    std::vector<T> expected(comm.size());
    for (int i = 0; i < comm.size(); ++i) {
        expected[i] = static_cast<T>(i);
    }
    check_single_all_gather<T>(rank, expected);
}

TEST(SimplemcMPI, AllGatherSingleValues) {
    // signed integer types
    perform_single<short>();
    perform_single<int>();
    perform_single<long>();
    perform_single<long long>();

    // unsigned integer types
    perform_single<unsigned short>();
    perform_single<unsigned int>();
    perform_single<unsigned long>();
    perform_single<unsigned long long>();

    // floating point types
    perform_single<float>();
    perform_single<double>();
    perform_single<long double>();
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single_complex() {
    simplemc::mpi::communicator comm {};
    const T rank = simplemc::mpi::comm_rank(comm);
    auto in = std::complex<T> { rank, -rank };
    std::vector<std::complex<T>> expected(comm.size());
    for (int i = 0; i < comm.size(); ++i) {
        expected[i] = std::complex<T> { static_cast<T>(i), static_cast<T>(-i) };
    }
    check_single_all_gather(in, expected);
}

TEST(SimplemcMPI, AllGatherSingleComplexValues) {
    perform_single_complex<float>();
    perform_single_complex<double>();
    perform_single_complex<long double>();
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single_in_place() {
    simplemc::mpi::communicator comm {};
    const T rank = simplemc::mpi::comm_rank(comm);
    std::vector<T> input(comm.size());
    input[comm.rank()] = rank;
    std::vector<T> expected(comm.size());
    for (int i = 0; i < comm.size(); ++i) {
        expected[i] = static_cast<T>(i);
    }
    check_single_all_gather_in_place<T>(input, expected);
}

TEST(SimplemcMPI, AllGatherInPlaceSingleValues) {
    // signed integer types
    perform_single_in_place<short>();
    perform_single_in_place<int>();
    perform_single_in_place<long>();
    perform_single_in_place<long long>();

    // unsigned integer types
    perform_single_in_place<unsigned short>();
    perform_single_in_place<unsigned int>();
    perform_single_in_place<unsigned long>();
    perform_single_in_place<unsigned long long>();

    // floating point types
    perform_single_in_place<float>();
    perform_single_in_place<double>();
    perform_single_in_place<long double>();
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single_in_place_complex() {
    simplemc::mpi::communicator comm {};
    const T rank = simplemc::mpi::comm_rank(comm);
    std::vector<std::complex<T>> input(comm.size());
    input[comm.rank()] = std::complex<T> { rank, -rank };
    std::vector<std::complex<T>> expected(comm.size());
    for (int i = 0; i < comm.size(); ++i) {
        expected[i] = std::complex<T> { static_cast<T>(i), static_cast<T>(-i) };
    }
    check_single_all_gather_in_place(input, expected);
}

TEST(SimplemcMPI, AllGatherInPlaceSingleComplexValues) {
    perform_single_in_place_complex<float>();
    perform_single_in_place_complex<double>();
    perform_single_in_place_complex<long double>();
}

TEST(SimplemcMPI, AllGatherEmptyVector) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_data {};
    std::vector<int> out_data {};
    simplemc::mpi::all_gather(in_data, out_data, comm);
    ASSERT_TRUE(out_data.empty());
}

// Helper function for performing repeated range tests.
template <typename T>
void perform_range() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const auto in = std::vector<T> { static_cast<T>(rank), static_cast<T>(2 * rank) };
    std::vector<T> expected;
    expected.reserve(2 * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        expected.push_back(static_cast<T>(i));
        expected.push_back(static_cast<T>(2 * i));
    }
    check_range_all_gather(in, expected);
}

TEST(SimplemcMPI, AllGatherRanges) {
    // signed integer types
    perform_range<short>();
    perform_range<int>();
    perform_range<long>();
    perform_range<long long>();

    // unsigned integer types
    perform_range<unsigned short>();
    perform_range<unsigned int>();
    perform_range<unsigned long>();
    perform_range<unsigned long long>();

    // floating point types
    perform_range<float>();
    perform_range<double>();
    perform_range<long double>();
}

// Helper function for performing repeated complex range tests.
template <typename T>
void perform_range_complex() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    auto in = std::vector<std::complex<T>> { { static_cast<T>(rank), static_cast<T>(2 * rank) },
        { static_cast<T>(3 * rank), static_cast<T>(4 * rank) } };
    std::vector<std::complex<T>> expected;
    expected.reserve(2 * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        expected.push_back({ static_cast<T>(i), static_cast<T>(2 * i) });
        expected.push_back({ static_cast<T>(3 * i), static_cast<T>(4 * i) });
    }
    check_range_all_gather(in, expected);
}

TEST(SimplemcMPI, AllGatherComplexRanges) {
    perform_range_complex<float>();
    perform_range_complex<double>();
    perform_range_complex<long double>();
}

// Helper function for performing repeated in-place range tests.
template <typename T>
void perform_range_in_place() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    std::vector<T> input(2 * comm_size, 0);
    std::vector<T> expected;
    for (int i = 0; i < comm_size; ++i) {
        expected.push_back(static_cast<T>(i));
        expected.push_back(static_cast<T>(static_cast<T>(2) * i));
        if (i == rank) {
            input[2 * i] = expected[2 * i];
            input[2 * i + 1] = expected[2 * i + 1];
        }
    }
    check_range_all_gather_in_place(input, expected);
}

TEST(SimplemcMPI, AllGatherInPlaceRanges) {
    // signed integer types
    perform_range_in_place<short>();
    perform_range_in_place<int>();
    perform_range_in_place<long>();
    perform_range_in_place<long long>();

    // unsigned integer types
    perform_range_in_place<unsigned short>();
    perform_range_in_place<unsigned int>();
    perform_range_in_place<unsigned long>();
    perform_range_in_place<unsigned long long>();

    // floating point types
    perform_range_in_place<float>();
    perform_range_in_place<double>();
    perform_range_in_place<long double>();
}

// Helper function for performing repeated complex in-place range tests.
template <typename T>
void perform_range_in_place_complex() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    std::vector<std::complex<T>> input(2 * comm_size, 0);
    std::vector<std::complex<T>> expected;
    for (int i = 0; i < comm_size; ++i) {
        expected.push_back({ static_cast<T>(i), static_cast<T>(static_cast<T>(2) * i) });
        expected.push_back({ static_cast<T>(static_cast<T>(3) * i), static_cast<T>(static_cast<T>(4) * i) });
        if (i == rank) {
            input[2 * i] = expected[2 * i];
            input[2 * i + 1] = expected[2 * i + 1];
        }
    }
    check_range_all_gather_in_place(input, expected);
}

TEST(SimplemcMPI, AllGatherInPlaceComplexRanges) {
    perform_range_in_place_complex<float>();
    perform_range_in_place_complex<double>();
    perform_range_in_place_complex<long double>();
}

TEST(SimplemcMPI, AllGatherWithSplitCommunicator) {
    simplemc::mpi::communicator comm {};
    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // split into even and odd ranks
    const int color = comm.rank() % 2;
    auto sub_comm = comm.split(color);

    const int rank = sub_comm.rank();
    const int size = sub_comm.size();

    // all-gather within sub-communicator
    std::vector<int> result(size);
    simplemc::mpi::all_gather(rank, result, sub_comm);
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(result[i], i);
    }

    // test in-place version
    std::vector<int> in_place_data(size, 0);
    in_place_data[rank] = rank;
    simplemc::mpi::all_gather_in_place(in_place_data, sub_comm);
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(in_place_data[i], i);
    }
}

TEST(SimplemcMPI, AllGatherStrings) {
    simplemc::mpi::communicator comm {};
    if (comm.size() > 10) {
        GTEST_SKIP() << "Test works only for <= 10 processes";
    }
    const std::string rank_str = fmt::format("Rank_{}", comm.rank());
    std::string expected {};
    for (int i = 0; i < comm.size(); ++i) {
        expected += fmt::format("Rank_{}", i);
    }
    std::string res(rank_str.size() * comm.size(), ' ');
    simplemc::mpi::all_gather(rank_str, res, comm);
    ASSERT_EQ(res, expected);
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
