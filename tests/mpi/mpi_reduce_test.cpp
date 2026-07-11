// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <array>
#include <complex>
#include <functional>
#include <vector>

// Test reduce for a single value of type T with a given operation.
template <typename T>
void check_single_reduce(const T& in, const T& exp, MPI_Op op) {
    using namespace simplemc::mpi;
    communicator comm {};

    // reduce overloads to test
    auto fvec = std::vector<std::function<void(T&, int)>> {};
    fvec.push_back([&](T& out, int root) { reduce(&in, &out, 1, mpi_type<T>::get(), op, root, comm); });
    fvec.push_back([&](T& out, int root) { reduce(&in, &out, 1, op, root, comm); });
    fvec.push_back([&](T& out, int root) { reduce(in, out, op, root, comm); });

    // perform reduce and check the result
    for (int r = 0; r < comm_size(comm); ++r) {
        for (auto fun : fvec) {
            T result {};
            fun(result, r);
            if (comm.rank() == r) {
                ASSERT_EQ(result, exp);
            }
        }
    }
}

// Test reduce_in_place for a single value of type T with a given operation.
template <typename T>
void check_single_reduce_in_place(const T& in, const T& exp, MPI_Op op) {
    using namespace simplemc::mpi;
    communicator comm {};

    // reduce_in_place overloads to test
    auto fvec = std::vector<std::function<void(T&, int)>> {};
    fvec.push_back([&](T& value, int root) { reduce_in_place(&value, 1, op, root, comm); });
    fvec.push_back([&](T& value, int root) { reduce_in_place(value, op, root, comm); });

    // perform reduce_in_place and check the result
    for (int r = 0; r < comm_size(comm); ++r) {
        for (auto fun : fvec) {
            T value = in;
            fun(value, r);
            if (comm.rank() == r) {
                ASSERT_EQ(value, exp);
            }
        }
    }
}

// Test reduce for a range of type R with a given operation.
template <typename R>
void check_range_reduce(const R& in, const R& exp, MPI_Op op) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const count = static_cast<int>(simplemc::ranges::size(in));
    auto mpi_dtype = mpi_type<value_type>::get();

    // reduce overloads to test
    auto fvec = std::vector<std::function<void(R&, int)>> {};
    fvec.push_back([&](R& out_rg, int root) {
        reduce(simplemc::ranges::data(in), simplemc::ranges::data(out_rg), count, mpi_dtype, op, root, comm);
    });
    fvec.push_back([&](R& out_rg, int root) {
        reduce(simplemc::ranges::data(in), simplemc::ranges::data(out_rg), count, op, root, comm);
    });
    fvec.push_back([&](R& out_rg, int root) { reduce(in, out_rg, op, root, comm); });

    // perform reduce and check the result
    for (int r = 0; r < comm_size(comm); ++r) {
        for (auto fun : fvec) {
            R result {};
            result.resize(count);
            fun(result, r);
            if (comm.rank() == r) {
                ASSERT_EQ(result, exp);
            }
        }
    }
}

// Test reduce_in_place for a range of type R with a given operation.
template <typename R>
void check_range_reduce_in_place(const R& in, const R& exp, MPI_Op op) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto const count = static_cast<int>(simplemc::ranges::size(in));

    // reduce_in_place overloads to test
    auto fvec = std::vector<std::function<void(R&, int)>> {};
    fvec.push_back([&](R& rg, int root) { reduce_in_place(simplemc::ranges::data(rg), count, op, root, comm); });
    fvec.push_back([&](R& rg, int root) { reduce_in_place(rg, op, root, comm); });

    // perform reduce_in_place and check the result
    for (int r = 0; r < comm_size(comm); ++r) {
        for (auto fun : fvec) {
            R data = in;
            fun(data, r);
            if (comm.rank() == r) {
                ASSERT_EQ(data, exp);
            }
        }
    }
}

// Helper function for performing tests on single values.
template <typename T>
void perform_single_value_test(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);

    // prepare input and expected output
    const int sum_ranks = (size * (size - 1)) / 2;
    T in = rank;
    T exp = sum_ranks;
    if constexpr (!std::is_arithmetic_v<T>) {
        in.imag(-rank);
        exp.imag(-sum_ranks);
    }

    // perform test
    if (in_place) {
        check_single_reduce_in_place<T>(in, exp, MPI_SUM);
    } else {
        check_single_reduce<T>(in, exp, MPI_SUM);
    }
}

// Helper function for performing tests on ranges.
template <typename T>
void perform_range_test(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int size = simplemc::mpi::comm_size(comm);
    const int rank = simplemc::mpi::comm_rank(comm);

    // prepare input and expected output
    const int sum_ranks = (size * (size - 1)) / 2;
    const int count = 5;
    std::vector<T> in, exp;
    for (int i = 0; i < count; ++i) {
        in.push_back(rank * (i + 1));
        exp.push_back(sum_ranks * (i + 1));
        if constexpr (!std::is_arithmetic_v<T>) {
            in.back().imag(-rank * (i + 1));
            exp.back().imag(-sum_ranks * (i + 1));
        }
    }

    // perform test
    if (in_place) {
        check_range_reduce_in_place(in, exp, MPI_SUM);
    } else {
        check_range_reduce(in, exp, MPI_SUM);
    }
}

TEST(SimplemcMPI, ReduceZeroValues) {
    simplemc::mpi::communicator comm {};
    int in_value = comm.rank();
    int out_value = -1;
    constexpr int root = 0;
    simplemc::mpi::reduce(&in_value, &out_value, 0, MPI_INT, MPI_SUM, root, comm);
    ASSERT_EQ(out_value, -1);
}

TEST(SimplemcMPI, ReduceSumSingleValues) {
    // signed integer types
    perform_single_value_test<short>();
    perform_single_value_test<int>();
    perform_single_value_test<long>();
    perform_single_value_test<long long>();

    // unsigned integer types
    perform_single_value_test<unsigned short>();
    perform_single_value_test<unsigned int>();
    perform_single_value_test<unsigned long>();
    perform_single_value_test<unsigned long long>();

    // floating point types
    perform_single_value_test<float>();
    perform_single_value_test<double>();
    perform_single_value_test<long double>();

    // complex types
    perform_single_value_test<std::complex<float>>();
    perform_single_value_test<std::complex<double>>();
    perform_single_value_test<std::complex<long double>>();
}

TEST(SimplemcMPI, ReduceSumInPlaceSingleValues) {
    // signed integer types
    perform_single_value_test<short>(true);
    perform_single_value_test<int>(true);
    perform_single_value_test<long>(true);
    perform_single_value_test<long long>(true);

    // unsigned integer types
    perform_single_value_test<unsigned short>(true);
    perform_single_value_test<unsigned int>(true);
    perform_single_value_test<unsigned long>(true);
    perform_single_value_test<unsigned long long>(true);

    // floating point types
    perform_single_value_test<float>(true);
    perform_single_value_test<double>(true);
    perform_single_value_test<long double>(true);

    // complex types
    perform_single_value_test<std::complex<float>>(true);
    perform_single_value_test<std::complex<double>>(true);
    perform_single_value_test<std::complex<long double>>(true);
}

TEST(SimplemcMPI, ReduceMinMaxSingleValues) {
    simplemc::mpi::communicator comm {};
    auto const size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    check_single_reduce<int>(rank, size - 1, MPI_MAX);
    check_single_reduce<double>(static_cast<double>(rank), static_cast<double>(size - 1), MPI_MAX);
    check_single_reduce<int>(rank, 0, MPI_MIN);
    check_single_reduce<double>(static_cast<double>(rank), 0.0, MPI_MIN);
}

TEST(SimplemcMPI, ReduceEmptyVector) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_data {};
    std::vector<int> out_data {};
    constexpr int root = 0;
    simplemc::mpi::reduce(in_data, out_data, MPI_SUM, root, comm);
    ASSERT_TRUE(out_data.empty());
}

TEST(SimplemcMPI, ReduceSumRanges) {
    // signed integer types
    perform_range_test<short>();
    perform_range_test<int>();
    perform_range_test<long>();
    perform_range_test<long long>();

    // unsigned integer types
    perform_range_test<unsigned short>();
    perform_range_test<unsigned int>();
    perform_range_test<unsigned long>();
    perform_range_test<unsigned long long>();

    // floating point types
    perform_range_test<float>();
    perform_range_test<double>();
    perform_range_test<long double>();

    // complex types
    perform_range_test<std::complex<float>>();
    perform_range_test<std::complex<double>>();
    perform_range_test<std::complex<long double>>();
}

TEST(SimplemcMPI, ReduceSumInPlaceRanges) {
    // signed integer types
    perform_range_test<short>(true);
    perform_range_test<int>(true);
    perform_range_test<long>(true);
    perform_range_test<long long>(true);

    // unsigned integer types
    perform_range_test<unsigned short>(true);
    perform_range_test<unsigned int>(true);
    perform_range_test<unsigned long>(true);
    perform_range_test<unsigned long long>(true);

    // floating point types
    perform_range_test<float>(true);
    perform_range_test<double>(true);
    perform_range_test<long double>(true);

    // complex types
    perform_range_test<std::complex<float>>(true);
    perform_range_test<std::complex<double>>(true);
    perform_range_test<std::complex<long double>>(true);
}

TEST(SimplemcMPI, ReduceMinMaxRanges) {
    simplemc::mpi::communicator comm {};
    auto const size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    std::vector<int> in = { 1 * (rank + 1), 2 * (rank + 1), 3 * (rank + 1) };
    std::vector<int> exp_max = { 1 * size, 2 * size, 3 * size };
    std::vector<int> exp_min = { 1 * 1, 2 * 1, 3 * 1 };
    check_range_reduce(in, exp_max, MPI_MAX);
    check_range_reduce(in, exp_min, MPI_MIN);
}

TEST(SimplemcMPI, ReduceLargeArray) {
    simplemc::mpi::communicator comm {};
    auto const size = simplemc::mpi::comm_size(comm);
    auto const rank = simplemc::mpi::comm_rank(comm);
    constexpr int count = 10000;
    const int root = 0;

    // fill array with rank value
    auto in_arr = std::array<int, count> {};
    simplemc::ranges::fill(in_arr, rank);

    // perform reduce with sum
    auto out_arr = std::array<int, count> {};
    simplemc::mpi::reduce(in_arr, out_arr, MPI_SUM, root, comm);

    // expected result: sum of all ranks (only valid on root)
    if (comm.rank() == root) {
        auto const sum_ranks = (size * (size - 1)) / 2;
        auto expected = std::array<int, count> {};
        simplemc::ranges::fill(expected, sum_ranks);
        ASSERT_EQ(out_arr, expected);
    }
}

TEST(SimplemcMPI, ReduceWithSplitCommunicator) {
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
    constexpr int root = 0;

    // reduce within sub-communicator
    int result = 0;
    simplemc::mpi::reduce(rank, result, MPI_SUM, root, sub_comm);
    if (sub_comm.rank() == root) {
        ASSERT_EQ(result, expected_sum);
    }

    // test in-place version
    int in_place_value = rank;
    simplemc::mpi::reduce_in_place(in_place_value, MPI_SUM, root, sub_comm);
    if (sub_comm.rank() == root) {
        ASSERT_EQ(in_place_value, expected_sum);
    }
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
