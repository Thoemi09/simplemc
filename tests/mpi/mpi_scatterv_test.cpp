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

// Test scatterv with varying counts per process.
template <typename T>
void check_scatterv(const std::vector<T>& in, const std::vector<T>& exp, int root) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto mpi_dtype = mpi_type<T>::get();
    auto [sendcounts, displs] = detail::prepare_scatterv_counts_displs(exp.size(), root, comm);

    // scatterv overloads to test
    auto fvec = std::vector<std::function<void(std::vector<T>&)>> {};
    fvec.push_back([&](std::vector<T>& out) {
        scatterv(in.data(), sendcounts.data(), displs.data(), mpi_dtype, out.data(), out.size(), mpi_dtype, root, comm);
    });
    fvec.push_back([&](std::vector<T>& out) {
        scatterv(in.data(), sendcounts.data(), displs.data(), out.data(), out.size(), root, comm);
    });
    fvec.push_back([&](std::vector<T>& out) { scatterv(in, out, root, comm); });

    // perform scatterv and check the result
    for (auto fun : fvec) {
        std::vector<T> result(exp.size());
        fun(result);
        ASSERT_EQ(result, exp);
    }
}

// Test scatterv_in_place with varying counts per process.
template <typename T>
void check_scatterv_in_place(const std::vector<T>& in, int local_count, const std::vector<T>& exp, int root) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto [sendcounts, displs] = detail::prepare_scatterv_counts_displs(local_count, root, comm);

    // scatterv_in_place overloads to test
    auto fvec = std::vector<std::function<void(std::vector<T>&)>> {};
    fvec.push_back([&](std::vector<T>& in_out) {
        scatterv_in_place(in_out.data(), sendcounts.data(), displs.data(), local_count, root, comm);
    });
    fvec.push_back([&](std::vector<T>& in_out) { scatterv_in_place(in_out, local_count, root, comm); });

    // perform scatterv_in_place and check the result
    for (auto fun : fvec) {
        std::vector<T> data = in;
        fun(data);
        ASSERT_EQ(data, exp);
    }
}

// Helper function for performing tests with varying counts per process.
template <typename T>
void perform_test(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();
    const int count = rank + 1;

    // prepare input (on root) and expected output (each process receives (rank + 1) elements)
    std::vector<T> in_root;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            in_root.push_back(r * 10 + i);
            if constexpr (!std::is_arithmetic_v<T>) {
                in_root.back().imag(-r * 10 - i);
            }
        }
    }

    // expected output for this rank
    const int disp = (rank * (rank + 1)) / 2;
    auto it = in_root.begin() + disp;
    std::vector<T> exp(it, it + count);

    // perform test for each rank as root
    for (int r = 0; r < size; ++r) {
        if (in_place) {
            if (rank == r) {
                check_scatterv_in_place<T>(in_root, count, in_root, r);
            } else {
                check_scatterv_in_place<T>(std::vector<T>(count, T {}), count, exp, r);
            }
        } else {
            check_scatterv(in_root, exp, r);
        }
    }
}

TEST(SimplemcMPI, ScattervZeroValues) {
    simplemc::mpi::communicator comm {};
    std::vector<int> out_rg {};
    constexpr int root = 0;
    simplemc::mpi::scatterv(std::vector<int> {}, out_rg, root, comm);
    ASSERT_TRUE(out_rg.empty());
}

TEST(SimplemcMPI, ScattervVaryingCounts) {
    // signed integer types
    perform_test<short>();
    perform_test<int>();
    perform_test<long>();
    perform_test<long long>();

    // unsigned integer types
    perform_test<unsigned short>();
    perform_test<unsigned int>();
    perform_test<unsigned long>();
    perform_test<unsigned long long>();

    // floating point types
    perform_test<float>();
    perform_test<double>();
    perform_test<long double>();

    // complex types
    perform_test<std::complex<float>>();
    perform_test<std::complex<double>>();
    perform_test<std::complex<long double>>();
}

TEST(SimplemcMPI, ScattervInPlaceVaryingCounts) {
    // signed integer types
    perform_test<short>(true);
    perform_test<int>(true);
    perform_test<long>(true);
    perform_test<long long>(true);

    // unsigned integer types
    perform_test<unsigned short>(true);
    perform_test<unsigned int>(true);
    perform_test<unsigned long>(true);
    perform_test<unsigned long long>(true);

    // floating point types
    perform_test<float>(true);
    perform_test<double>(true);
    perform_test<long double>(true);

    // complex types
    perform_test<std::complex<float>>(true);
    perform_test<std::complex<double>>(true);
    perform_test<std::complex<long double>>(true);
}

TEST(SimplemcMPI, ScattervUniformCounts) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    const int count = 3;

    // prepare input data (on root) - will be scattered
    std::vector<int> in_root;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < count; ++i) {
            in_root.push_back(r * 100 + i);
        }
    }

    // expected output for this rank
    auto it = in_root.begin() + static_cast<long>(rank) * count;
    std::vector<int> exp(it, it + count);

    // scatterv and check result for each root
    for (int r = 0; r < size; ++r) {
        std::vector<int> result(count);
        simplemc::mpi::scatterv(in_root, result, r, comm);
        ASSERT_EQ(result, exp);
    }
}

TEST(SimplemcMPI, ScattervWithSplitCommunicator) {
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

    // each process receives (rank + 1) elements
    std::vector<int> in_root, exp;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            in_root.push_back(r * 100 + i);
            if (r == rank) {
                exp.push_back(in_root.back());
            }
        }
    }

    // scatterv and check result
    std::vector<int> result(exp.size());
    simplemc::mpi::scatterv(in_root, result, root, sub_comm);
    ASSERT_EQ(result, exp);
}

TEST(SimplemcMPI, ScattervEmptyToSomeProcesses) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    constexpr int root = 0;

    // only even ranks receive data
    std::vector<double> in_root, exp;
    for (int r = 0; r < size; ++r) {
        if (r % 2 == 0) {
            for (int i = 0; i < 2; ++i) {
                in_root.push_back(static_cast<double>(r * 10 + i));
                if (r == rank) {
                    exp.push_back(in_root.back());
                }
            }
        }
    }

    // scatterv and check result
    std::vector<double> result(exp.size());
    simplemc::mpi::scatterv(in_root, result, root, comm);
    ASSERT_EQ(result, exp);
}

TEST(SimplemcMPI, ScattervStrings) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    if (size > 10) {
        GTEST_SKIP() << "Test works only for <= 10 processes";
    }

    // each process receives a string with length equal to (rank + 1)
    std::string in_str {};
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            in_str += fmt::format("{}", r);
        }
    }
    std::string expected(rank + 1, static_cast<char>('0' + rank));

    // scatterv and check result for each root
    for (int r = 0; r < size; ++r) {
        std::string result(rank + 1, ' ');
        simplemc::mpi::scatterv(in_str, result, r, comm);
        ASSERT_EQ(result, expected);
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
