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

// Test scatter for a single value of type T.
template <typename T>
void check_single_scatter(const std::vector<T>& in, const T& exp, int root) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto mpi_dtype = mpi_type<T>::get();

    // scatter overloads to test
    auto fvec = std::vector<std::function<void(T&)>> {};
    fvec.push_back([&](T& out) { scatter(in.data(), 1, mpi_dtype, &out, 1, mpi_dtype, root, comm); });
    fvec.push_back([&](T& out) { scatter(in.data(), &out, 1, root, comm); });
    fvec.push_back([&](T& out) { scatter(in, out, root, comm); });

    // perform scatter and check the result
    for (auto fun : fvec) {
        T result {};
        fun(result);
        ASSERT_EQ(result, exp);
    }
}

// Test scatter for a range of type R.
template <typename R>
void check_range_scatter(const R& in, const R& exp, int root) {
    using namespace simplemc::mpi;
    using value_type = simplemc::ranges::range_value_t<R>;
    communicator comm {};
    auto const count = simplemc::ranges::size(exp);
    auto mpi_dtype = mpi_type<value_type>::get();

    // scatter overloads to test
    auto fvec = std::vector<std::function<void(R&)>> {};
    fvec.push_back([&](R& out_rg) {
        scatter(
            simplemc::ranges::data(in), count, mpi_dtype, simplemc::ranges::data(out_rg), count, mpi_dtype, root, comm);
    });
    fvec.push_back(
        [&](R& out_rg) { scatter(simplemc::ranges::data(in), simplemc::ranges::data(out_rg), count, root, comm); });
    fvec.push_back([&](R& out_rg) { scatter(in, out_rg, root, comm); });

    // perform scatter and check the result
    for (auto fun : fvec) {
        R result {};
        result.resize(simplemc::ranges::size(exp));
        fun(result);
        ASSERT_EQ(result, exp);
    }
}

// Test scatter_in_place for a range of type R.
// Note: in and exp must have the correct sizes based on rank (root: count*size, non-root: count).
template <typename R>
void check_range_scatter_in_place(const R& in, const R& exp, int root) {
    using namespace simplemc::mpi;
    communicator comm {};

    // determine count per process (root has full data, non-root has per-process portion)
    auto count = simplemc::ranges::size(in);
    if (comm.rank() == root) {
        count /= comm.size();
    }

    // scatter_in_place overloads to test
    auto fvec = std::vector<std::function<void(R&)>> {};
    fvec.push_back([&](R& rg) { scatter_in_place(simplemc::ranges::data(rg), count, root, comm); });
    fvec.push_back([&](R& rg) { scatter_in_place(rg, root, comm); });

    // perform scatter_in_place and check the result
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

    // prepare input (on root) and expected output (on each rank)
    std::vector<T> in(size);
    for (int r = 0; r < size; ++r) {
        in[r] = r;
        if constexpr (!std::is_arithmetic_v<T>) {
            in[r].imag(-r);
        }
    }
    T exp = in[rank];

    // perform test for each rank as root
    for (int r = 0; r < size; ++r) {
        check_single_scatter<T>(in, exp, r);
    }
}

// Helper function for performing tests on ranges.
template <typename T>
void perform_range_test(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();
    const int count = 3;

    // prepare input data (on root) - will be scattered
    std::vector<T> in_root;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < count; ++i) {
            in_root.push_back(10 * r + i);
            if constexpr (!std::is_arithmetic_v<T>) {
                in_root.back().imag(-10 * r - i);
            }
        }
    }

    // expected output for this rank
    auto it = in_root.begin() + rank * count;
    std::vector<T> exp(it, it + count);

    // perform test for each rank as root
    for (int r = 0; r < size; ++r) {
        if (in_place) {
            if (rank == r) {
                check_range_scatter_in_place(in_root, in_root, r);
            } else {
                check_range_scatter_in_place(std::vector<T>(count, T(0)), exp, r);
            }
        } else {
            check_range_scatter(in_root, exp, r);
        }
    }
}

TEST(SimplemcMPI, ScatterZeroValues) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_values(comm.size(), comm.rank());
    int out_value = -1;
    constexpr int root = 0;
    simplemc::mpi::scatter(in_values.data(), &out_value, 0, root, comm);
    ASSERT_EQ(out_value, -1);
}

TEST(SimplemcMPI, ScatterSingleValues) {
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

TEST(SimplemcMPI, ScatterEmptyVector) {
    simplemc::mpi::communicator comm {};
    std::vector<int> in_data {};
    std::vector<int> out_data {};
    constexpr int root = 0;
    simplemc::mpi::scatter(in_data, out_data, root, comm);
    ASSERT_TRUE(out_data.empty());
}

TEST(SimplemcMPI, ScatterRanges) {
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

TEST(SimplemcMPI, ScatterInPlaceRanges) {
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

TEST(SimplemcMPI, ScatterWithSplitCommunicator) {
    simplemc::mpi::communicator comm {};
    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // split into even and odd ranks
    const int color = comm.rank() % 2;
    auto sub_comm = comm.split(color);

    const int rank = sub_comm.rank();
    const int size = sub_comm.size();
    constexpr int root = 0;

    // prepare input data (on root)
    std::vector<int> in_data(size);
    for (int r = 0; r < size; ++r) {
        in_data[r] = r * 10;
    }

    // scatter and check result
    int result = -1;
    simplemc::mpi::scatter(in_data, result, root, sub_comm);
    ASSERT_EQ(result, rank * 10);
}

TEST(SimplemcMPI, ScatterStrings) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();
    if (size > 10) {
        GTEST_SKIP() << "Test works only for <= 10 processes";
    }

    // prepare input string (on root) and expected output for this rank
    std::string in_str {};
    for (int r = 0; r < size; ++r) {
        in_str += fmt::format("Rank_{}", r);
    }
    const std::string exp = fmt::format("Rank_{}", rank);

    // scatter and check result for each root
    for (int r = 0; r < size; ++r) {
        std::string res(exp.size(), ' ');
        simplemc::mpi::scatter(in_str, res, r, comm);
        ASSERT_EQ(res, exp);
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
