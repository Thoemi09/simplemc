// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

// Test all_gather for a single value of type T.
template <typename T>
void check_single_all_gather(const T& in, const std::vector<T>& exp) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto mpi_dtype = mpi_type<T>::get();

    // all_gather overloads to test
    auto fvec = std::vector<std::function<void(std::vector<T>&)>> {};
    fvec.push_back([&](std::vector<T>& out) { all_gather(&in, 1, mpi_dtype, out.data(), 1, mpi_dtype, comm); });
    fvec.push_back([&](std::vector<T>& out) { all_gather(&in, out.data(), 1, comm); });
    fvec.push_back([&](std::vector<T>& out) { all_gather(in, out, comm); });

    // perform all_gather and check the result
    for (auto fun : fvec) {
        std::vector<T> result(comm.size());
        fun(result);
        ASSERT_EQ(result, exp);
    }
}

// Test all_gather for a range of type R.
template <typename R>
void check_range_all_gather(const R& in, const R& exp) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const count = simplemc::ranges::size(in);
    auto mpi_dtype = mpi_type<value_type>::get();

    // all_gather overloads to test
    auto fvec = std::vector<std::function<void(R&)>> {};
    fvec.push_back([&](R& out_rg) {
        all_gather(
            simplemc::ranges::data(in), count, mpi_dtype, simplemc::ranges::data(out_rg), count, mpi_dtype, comm);
    });
    fvec.push_back(
        [&](R& out_rg) { all_gather(simplemc::ranges::data(in), simplemc::ranges::data(out_rg), count, comm); });
    fvec.push_back([&](R& out_rg) { all_gather(in, out_rg, comm); });

    // perform all_gather and check the result
    for (auto fun : fvec) {
        R result {};
        result.resize(simplemc::ranges::size(exp));
        fun(result);
        ASSERT_EQ(result, exp);
    }
}

// Test all_gather_in_place for a range of type R.
template <typename R>
void check_range_all_gather_in_place(const R& in, const R& exp) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto const total_count = simplemc::ranges::size(in);
    auto const count = total_count / comm.size();

    // all_gather_in_place overloads to test
    auto fvec = std::vector<std::function<void(R&)>> {};
    fvec.push_back([&](R& rg) { all_gather_in_place(simplemc::ranges::data(rg), count, comm); });
    fvec.push_back([&](R& rg) { all_gather_in_place(rg, comm); });

    // perform all_gather_in_place and check the result
    for (auto fun : fvec) {
        R data = in;
        fun(data);
        ASSERT_EQ(data, exp);
    }
}

// Helper function for performing tests on single values.
template <typename T>
void perform_single_value_test() {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();

    // prepare input and expected output
    std::vector<T> in, exp;
    for (int r = 0; r < size; ++r) {
        exp.push_back(r);
        if constexpr (!std::is_arithmetic_v<T>) {
            exp.back().imag(-r);
        }
        in.push_back((r == rank ? exp.back() : T(0)));
    }

    // perform test
    check_single_all_gather<T>(in[rank], exp);
}

// Helper function for performing tests on ranges.
template <typename T>
void perform_range_test(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();
    const int count = 3;

    // prepare input and expected output
    std::vector<T> in, exp;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < count; ++i) {
            exp.push_back(10 * r + i);
            if constexpr (!std::is_arithmetic_v<T>) {
                exp.back().imag(-10 * r - i);
            }
            in.push_back((r == rank ? exp.back() : T(0)));
        }
    }

    // perform test
    if (in_place) {
        check_range_all_gather_in_place(in, exp);
    } else {
        auto it = in.begin() + count * rank;
        check_range_all_gather(std::vector<T>(it, it + count), exp);
    }
}

TEST(SimplemcMPI, AllGatherZeroValues) {
    simplemc::mpi::communicator comm {};
    const int in_value = comm.rank();
    std::vector<int> out_values(comm.size(), -1);
    simplemc::mpi::all_gather(&in_value, out_values.data(), 0, comm);
    for (int r = 0; r < comm.size(); ++r) {
        ASSERT_EQ(out_values[r], -1);
    }
}

TEST(SimplemcMPI, AllGatherSingleValues) {
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

TEST(SimplemcMPI, AllGatherEmptyVector) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_data {};
    std::vector<int> out_data {};
    simplemc::mpi::all_gather(in_data, out_data, comm);
    ASSERT_TRUE(out_data.empty());
}

TEST(SimplemcMPI, AllGatherRanges) {
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

TEST(SimplemcMPI, AllGatherInPlaceRanges) {
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

    // allgatherv and check result
    std::vector<int> result(size);
    simplemc::mpi::all_gather(rank, result, sub_comm);
    for (int r = 0; r < size; ++r) {
        ASSERT_EQ(result[r], r);
    }

    // allgatherv in place and check result
    std::vector<int> in_place_data(size, 0);
    in_place_data[rank] = rank;
    simplemc::mpi::all_gather_in_place(in_place_data, sub_comm);
    for (int r = 0; r < size; ++r) {
        ASSERT_EQ(in_place_data[r], r);
    }
}

TEST(SimplemcMPI, AllGatherStrings) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();
    if (size > 10) {
        GTEST_SKIP() << "Test works only for <= 10 processes";
    }

    // prepare input and expected output string
    const std::string rank_str = fmt::format("Rank_{}", rank);
    std::string exp {};
    for (int r = 0; r < size; ++r) {
        exp += fmt::format("Rank_{}", r);
    }

    // allgather and check result
    std::string res(rank_str.size() * size, ' ');
    simplemc::mpi::all_gather(rank_str, res, comm);
    ASSERT_EQ(res, exp);
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
