#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <array>
#include <complex>
#include <functional>
#include <vector>

// Test all-reducing a single value of type T with a given operation.
template <typename T>
void check_single_all_reduce(const T& input, const T& expected, MPI_Op op) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto fvec = std::vector<std::function<void(const T&, T&)>> {};
    fvec.push_back([&](const T& in, T& out) { all_reduce(&in, &out, 1, mpi_type<T>::get(), op, comm); });
    fvec.push_back([&](const T& in, T& out) { all_reduce(&in, &out, 1, op, comm); });
    fvec.push_back([&](const T& in, T& out) { all_reduce(in, out, op, comm); });
    for (auto all_reduce : fvec) {
        T result {};
        all_reduce(input, result);
        ASSERT_EQ(result, expected);
    }
}

// Test all-reducing a single value in place of type T with a given operation.
template <typename T>
void check_single_all_reduce_in_place(const T& input, const T& expected, MPI_Op op) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto fvec = std::vector<std::function<void(T&)>> {};
    fvec.push_back([&](T& value) { all_reduce_in_place(&value, 1, mpi_type<T>::get(), op, comm); });
    fvec.push_back([&](T& value) { all_reduce_in_place(&value, 1, op, comm); });
    fvec.push_back([&](T& value) { all_reduce_in_place(value, op, comm); });
    for (auto all_reduce_in_place : fvec) {
        T value = input;
        all_reduce_in_place(value);
        ASSERT_EQ(value, expected);
    }
}

// Test all-reducing a range of type R with a given operation.
template <typename R>
void check_range_all_reduce(const R& input, const R& expected, MPI_Op op) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const size = simplemc::ranges::size(input);
    auto mpi_dtype = mpi_type<value_type>::get();
    auto fvec = std::vector<std::function<void(const R&, R&)>> {};
    fvec.push_back([&](const R& in_rg, R& out_rg) {
        all_reduce(simplemc::ranges::data(in_rg), simplemc::ranges::data(out_rg), size, mpi_dtype, op, comm);
    });
    fvec.push_back([&](const R& in_rg, R& out_rg) {
        all_reduce(simplemc::ranges::data(in_rg), simplemc::ranges::data(out_rg), size, op, comm);
    });
    fvec.push_back([&](const R& in_rg, R& out_rg) { all_reduce(in_rg, out_rg, op, comm); });
    for (auto all_reduce : fvec) {
        R result {};
        result.resize(size);
        all_reduce(input, result);
        ASSERT_EQ(result, expected);
    }
}

// Test all-reducing a range in place of type R with a given operation.
template <typename R>
void check_range_all_reduce_in_place(const R& input, const R& expected, MPI_Op op) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const size = simplemc::ranges::size(input);
    auto mpi_dtype = mpi_type<value_type>::get();
    auto fvec = std::vector<std::function<void(R&)>> {};
    fvec.push_back([&](R& rg) { all_reduce_in_place(simplemc::ranges::data(rg), size, mpi_dtype, op, comm); });
    fvec.push_back([&](R& rg) { all_reduce_in_place(simplemc::ranges::data(rg), size, op, comm); });
    fvec.push_back([&](R& rg) { all_reduce_in_place(rg, op, comm); });
    for (auto all_reduce_in_place : fvec) {
        R data = input;
        all_reduce_in_place(data);
        ASSERT_EQ(data, expected);
    }
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int sum_ranks = (comm_size * (comm_size - 1)) / 2;
    T in = rank;
    T exp = sum_ranks;
    if constexpr (!std::is_arithmetic_v<T>) {
        in.imag(-rank);
        exp.imag(-sum_ranks);
    }
    if (in_place) {
        check_single_all_reduce_in_place<T>(in, exp, MPI_SUM);
    } else {
        check_single_all_reduce<T>(in, exp, MPI_SUM);
    }
}

// Helper function for performing repeated range tests.
template <typename T>
void perform_range(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int sum_ranks = (comm_size * (comm_size - 1)) / 2;
    const int sz = 5;
    std::vector<T> in, exp;
    for (int i = 0; i < sz; ++i) {
        in.push_back(rank * (i + 1));
        exp.push_back(sum_ranks * (i + 1));
        if constexpr (!std::is_arithmetic_v<T>) {
            in.back().imag(-rank * (i + 1));
            exp.back().imag(-sum_ranks * (i + 1));
        }
    }
    if (in_place) {
        check_range_all_reduce_in_place(in, exp, MPI_SUM);
    } else {
        check_range_all_reduce(in, exp, MPI_SUM);
    }
}

TEST(SimplemcMPI, AllReduceZeroValues) {
    simplemc::mpi::communicator comm {};
    int in_value = comm.rank();
    int out_value = -1;
    simplemc::mpi::all_reduce(&in_value, &out_value, 0, MPI_INT, MPI_SUM, comm);
    ASSERT_EQ(out_value, -1);
}

TEST(SimplemcMPI, AllReduceSumSingleValues) {
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

    // complex types
    perform_single<std::complex<float>>();
    perform_single<std::complex<double>>();
    perform_single<std::complex<long double>>();
}

TEST(SimplemcMPI, AllReduceSumInPlaceSingleValues) {
    // signed integer types
    perform_single<short>(true);
    perform_single<int>(true);
    perform_single<long>(true);
    perform_single<long long>(true);

    // unsigned integer types
    perform_single<unsigned short>(true);
    perform_single<unsigned int>(true);
    perform_single<unsigned long>(true);
    perform_single<unsigned long long>(true);

    // floating point types
    perform_single<float>(true);
    perform_single<double>(true);
    perform_single<long double>(true);

    // complex types
    perform_single<std::complex<float>>(true);
    perform_single<std::complex<double>>(true);
    perform_single<std::complex<long double>>(true);
}

TEST(SimplemcMPI, AllReduceMinMaxSingleValues) {
    simplemc::mpi::communicator comm {};
    auto const comm_size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    check_single_all_reduce<int>(rank, comm_size - 1, MPI_MAX);
    check_single_all_reduce<double>(static_cast<double>(rank), static_cast<double>(comm_size - 1), MPI_MAX);
    check_single_all_reduce<int>(rank, 0, MPI_MIN);
    check_single_all_reduce<double>(static_cast<double>(rank), 0.0, MPI_MIN);
}

TEST(SimplemcMPI, AllReduceEmptyVector) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_data {};
    std::vector<int> out_data {};
    simplemc::mpi::all_reduce(in_data, out_data, MPI_SUM, comm);
    ASSERT_TRUE(out_data.empty());
}

TEST(SimplemcMPI, AllReduceSumRanges) {
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

    // complex types
    perform_range<std::complex<float>>();
    perform_range<std::complex<double>>();
    perform_range<std::complex<long double>>();
}

TEST(SimplemcMPI, AllReduceSumInPlaceRanges) {
    // signed integer types
    perform_range<short>(true);
    perform_range<int>(true);
    perform_range<long>(true);
    perform_range<long long>(true);

    // unsigned integer types
    perform_range<unsigned short>(true);
    perform_range<unsigned int>(true);
    perform_range<unsigned long>(true);
    perform_range<unsigned long long>(true);

    // floating point types
    perform_range<float>(true);
    perform_range<double>(true);
    perform_range<long double>(true);

    // complex types
    perform_range<std::complex<float>>(true);
    perform_range<std::complex<double>>(true);
    perform_range<std::complex<long double>>(true);
}

TEST(SimplemcMPI, ReduceMinMaxSingleValues) {
    simplemc::mpi::communicator comm {};
    auto const comm_size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    check_single_all_reduce<int>(rank, comm_size - 1, MPI_MAX);
    check_single_all_reduce<double>(static_cast<double>(rank), static_cast<double>(comm_size - 1), MPI_MAX);
    check_single_all_reduce<int>(rank, 0, MPI_MIN);
    check_single_all_reduce<double>(static_cast<double>(rank), 0.0, MPI_MIN);
}

TEST(SimplemcMPI, AllReduceLargeArray) {
    simplemc::mpi::communicator comm {};
    constexpr int size = 10000;
    auto const comm_size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);

    // fill array with rank value
    auto in_arr = std::array<int, size> {};
    simplemc::ranges::fill(in_arr, rank);

    // perform allreduce with sum
    auto out_arr = std::array<int, size> {};
    simplemc::mpi::all_reduce(in_arr, out_arr, MPI_SUM, comm);

    // expected result: sum of all ranks
    auto const sum_ranks = (comm_size * (comm_size - 1)) / 2;
    auto expected = std::array<int, size> {};
    simplemc::ranges::fill(expected, sum_ranks);
    ASSERT_EQ(out_arr, expected);
}

TEST(SimplemcMPI, AllReduceWithSplitCommunicator) {
    simplemc::mpi::communicator comm {};
    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // split into even and odd ranks
    const int color = comm.rank() % 2;
    auto sub_comm = comm.split(color);

    // compute sum of ranks in each sub-communicator
    const int rank = sub_comm.rank();
    const int size = sub_comm.size();
    const int expected_sum = (size * (size - 1)) / 2;

    // all-reduce within sub-communicator
    int result = 0;
    simplemc::mpi::all_reduce(rank, result, MPI_SUM, sub_comm);
    ASSERT_EQ(result, expected_sum);

    // test in-place version
    int in_place_value = rank;
    simplemc::mpi::all_reduce_in_place(in_place_value, MPI_SUM, sub_comm);
    ASSERT_EQ(in_place_value, expected_sum);
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
