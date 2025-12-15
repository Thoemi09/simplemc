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

TEST(SimplemcMPI, AllReduceZeroValues) {
    simplemc::mpi::communicator comm {};
    int in_value = comm.rank();
    int out_value = -1;
    simplemc::mpi::all_reduce(&in_value, &out_value, 0, MPI_INT, MPI_SUM, comm);
    ASSERT_EQ(out_value, -1);
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single() {
    simplemc::mpi::communicator comm {};
    const T comm_size = simplemc::mpi::comm_size(comm);
    const T rank = simplemc::mpi::comm_rank(comm);
    check_single_all_reduce<T>(rank, comm_size * (comm_size - 1) / 2, MPI_SUM);
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
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single_complex() {
    simplemc::mpi::communicator comm {};
    const T comm_size = simplemc::mpi::comm_size(comm);
    const T rank = simplemc::mpi::comm_rank(comm);
    auto in = std::complex<T> { rank, -rank };
    auto exp = std::complex<T> { (comm_size * (comm_size - 1)) / 2, -(comm_size * (comm_size - 1)) / 2 };
    check_single_all_reduce(in, exp, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumSingleComplexValues) {
    perform_single_complex<float>();
    perform_single_complex<double>();
    perform_single_complex<long double>();
}

// Helper function for performing repeated tests.
template <typename T>
void perform_single_in_place() {
    simplemc::mpi::communicator comm {};
    const T comm_size = simplemc::mpi::comm_size(comm);
    const T rank = simplemc::mpi::comm_rank(comm);
    check_single_all_reduce_in_place<T>(rank, comm_size * (comm_size - 1) / 2, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumInPlaceSingleValues) {
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
    const T comm_size = simplemc::mpi::comm_size(comm);
    const T rank = simplemc::mpi::comm_rank(comm);
    auto in = std::complex<T> { rank, -rank };
    auto exp = std::complex<T> { (comm_size * (comm_size - 1)) / 2, -(comm_size * (comm_size - 1)) / 2 };
    check_single_all_reduce_in_place(in, exp, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumInPlaceSingleComplexValues) {
    perform_single_in_place_complex<float>();
    perform_single_in_place_complex<double>();
    perform_single_in_place_complex<long double>();
}

TEST(SimplemcMPI, AllReduceMaxSingleValues) {
    simplemc::mpi::communicator comm {};
    auto const comm_size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    auto const expected = comm_size - 1;
    check_single_all_reduce<int>(rank, expected, MPI_MAX);
    check_single_all_reduce<double>(static_cast<double>(rank), static_cast<double>(expected), MPI_MAX);
}

TEST(SimplemcMPI, AllReduceMinSingleValues) {
    simplemc::mpi::communicator comm {};
    auto const rank = simplemc::mpi::comm_rank(comm);
    constexpr int expected = 0;
    check_single_all_reduce<long>(rank, expected, MPI_MIN);
    check_single_all_reduce<float>(static_cast<float>(rank), static_cast<float>(expected), MPI_MIN);
}

TEST(SimplemcMPI, AllReduceProdSingleValues) {
    simplemc::mpi::communicator comm {};
    auto const comm_size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    long long expected = 1;
    for (int i = 1; i < comm_size; ++i) {
        expected *= i;
    }
    check_single_all_reduce<long long>(rank == 0 ? 1LL : rank, expected, MPI_PROD);
    check_single_all_reduce<long double>(
        rank == 0 ? 1.0L : static_cast<long double>(rank), static_cast<long double>(expected), MPI_PROD);
}

TEST(SimplemcMPI, AllReduceEmptyVector) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_data {};
    std::vector<int> out_data {};
    simplemc::mpi::all_reduce(in_data, out_data, MPI_SUM, comm);
    ASSERT_TRUE(out_data.empty());
}

// Helper function for performing repeated range tests.
template <typename T>
void perform_range_sum() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int sum_ranks = (comm_size * (comm_size - 1)) / 2;
    const auto in = std::vector<T> { static_cast<T>(rank), static_cast<T>(2 * rank) };
    const auto exp = std::vector<T> { static_cast<T>(sum_ranks), static_cast<T>(2 * sum_ranks) };
    check_range_all_reduce(in, exp, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumRanges) {
    // signed integer types
    perform_range_sum<short>();
    perform_range_sum<int>();
    perform_range_sum<long>();
    perform_range_sum<long long>();

    // unsigned integer types
    perform_range_sum<unsigned short>();
    perform_range_sum<unsigned int>();
    perform_range_sum<unsigned long>();
    perform_range_sum<unsigned long long>();

    // floating point types
    perform_range_sum<float>();
    perform_range_sum<double>();
    perform_range_sum<long double>();
}

// Helper function for performing repeated complex range tests.
template <typename T>
void perform_range_sum_complex() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int sum_ranks = (comm_size * (comm_size - 1)) / 2;
    auto in = std::vector<std::complex<T>> { { static_cast<T>(rank), static_cast<T>(2 * rank) },
        { static_cast<T>(3 * rank), static_cast<T>(4 * rank) } };
    auto exp = std::vector<std::complex<T>> { { static_cast<T>(sum_ranks), static_cast<T>(2 * sum_ranks) },
        { static_cast<T>(3 * sum_ranks), static_cast<T>(4 * sum_ranks) } };
    check_range_all_reduce(in, exp, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumComplexRanges) {
    perform_range_sum_complex<float>();
    perform_range_sum_complex<double>();
    perform_range_sum_complex<long double>();
}

// Helper function for performing repeated in-place range tests.
template <typename T>
void perform_range_sum_in_place() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int sum_ranks = (comm_size * (comm_size - 1)) / 2;
    const auto in = std::vector<T> { static_cast<T>(rank), static_cast<T>(2 * rank) };
    const auto exp = std::vector<T> { static_cast<T>(sum_ranks), static_cast<T>(2 * sum_ranks) };
    check_range_all_reduce_in_place(in, exp, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumInPlaceRanges) {
    // signed integer types
    perform_range_sum_in_place<short>();
    perform_range_sum_in_place<int>();
    perform_range_sum_in_place<long>();
    perform_range_sum_in_place<long long>();

    // unsigned integer types
    perform_range_sum_in_place<unsigned short>();
    perform_range_sum_in_place<unsigned int>();
    perform_range_sum_in_place<unsigned long>();
    perform_range_sum_in_place<unsigned long long>();

    // floating point types
    perform_range_sum_in_place<float>();
    perform_range_sum_in_place<double>();
    perform_range_sum_in_place<long double>();
}

// Helper function for performing repeated complex in-place range tests.
template <typename T>
void perform_range_sum_in_place_complex() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int sum_ranks = (comm_size * (comm_size - 1)) / 2;
    auto in = std::vector<std::complex<T>> { { static_cast<T>(rank), static_cast<T>(2 * rank) },
        { static_cast<T>(3 * rank), static_cast<T>(4 * rank) } };
    auto exp = std::vector<std::complex<T>> { { static_cast<T>(sum_ranks), static_cast<T>(2 * sum_ranks) },
        { static_cast<T>(3 * sum_ranks), static_cast<T>(4 * sum_ranks) } };
    check_range_all_reduce_in_place(in, exp, MPI_SUM);
}

TEST(SimplemcMPI, AllReduceSumInPlaceComplexRanges) {
    perform_range_sum_in_place_complex<float>();
    perform_range_sum_in_place_complex<double>();
    perform_range_sum_in_place_complex<long double>();
}

// Helper function for performing max range tests.
template <typename T>
void perform_range_max() {
    simplemc::mpi::communicator comm {};
    const int comm_size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);
    const int max_rank = comm_size - 1;
    check_range_all_reduce(std::vector<T> { static_cast<T>(rank), static_cast<T>(2 * rank) },
        std::vector<T> { static_cast<T>(max_rank), static_cast<T>(2 * max_rank) }, MPI_MAX);
}

TEST(SimplemcMPI, AllReduceMaxRanges) {
    perform_range_max<int>();
    perform_range_max<unsigned int>();
    perform_range_max<double>();
}

// Helper function for performing min range tests.
template <typename T>
void perform_range_min() {
    simplemc::mpi::communicator comm {};
    const int rank = simplemc::mpi::comm_rank(comm);
    constexpr int min_rank = 0;
    check_range_all_reduce(std::vector<T> { static_cast<T>(rank), static_cast<T>(2 * rank) },
        std::vector<T> { static_cast<T>(min_rank), static_cast<T>(2 * min_rank) }, MPI_MIN);
}

TEST(SimplemcMPI, AllReduceMinRanges) {
    perform_range_min<int>();
    perform_range_min<unsigned int>();
    perform_range_min<double>();
}

TEST(SimplemcMPI, AllReduceLargeArray) {
    simplemc::mpi::communicator comm {};
    constexpr int size = 10000;
    auto const comm_size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);

    // fill array with rank value
    auto in_arr = std::array<int, size> {};
    simplemc::ranges::fill(in_arr, rank);

    // perform all-reduce with sum
    auto out_arr = std::array<int, size> {};
    simplemc::mpi::all_reduce(in_arr, out_arr, MPI_SUM, comm);

    // expected result: sum of all ranks
    auto const sum_ranks = (comm_size * (comm_size - 1)) / 2;
    auto expected = std::array<int, size> {};
    simplemc::ranges::fill(expected, sum_ranks);

    check_range_equal(out_arr, expected);
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
