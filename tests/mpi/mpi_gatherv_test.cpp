#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

// Test gatherv with varying counts per process.
template <typename T>
void check_gatherv(const std::vector<T>& in, const std::vector<T>& exp, int root) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto mpi_dtype = mpi_type<T>::get();
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(in.size(), root, comm);

    // gatherv overloads to test
    auto fvec = std::vector<std::function<void(std::vector<T>&)>> {};
    fvec.push_back([&](std::vector<T>& out) {
        gatherv(in.data(), in.size(), mpi_dtype, out.data(), recvcounts.data(), displs.data(), mpi_dtype, root, comm);
    });
    fvec.push_back([&](std::vector<T>& out) {
        gatherv(in.data(), in.size(), out.data(), recvcounts.data(), displs.data(), root, comm);
    });
    fvec.push_back([&](std::vector<T>& out) { gatherv(in, out, root, comm); });

    // perform gatherv and check the result
    for (auto fun : fvec) {
        std::vector<T> result(exp.size());
        fun(result);
        if (comm.rank() == root) {
            ASSERT_EQ(result, exp);
        }
    }
}

// Test gatherv_in_place with varying counts per process.
template <typename T>
void check_gatherv_in_place(const std::vector<T>& in_root, const std::vector<T>& in_nonroot, int local_count,
    const std::vector<T>& exp, int root) {
    using namespace simplemc::mpi;
    communicator comm {};
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(local_count, root, comm);

    // gatherv_in_place overloads to test
    auto fvec = std::vector<std::function<void(std::vector<T>&)>> {};
    fvec.push_back([&](std::vector<T>& in_out) {
        gatherv_in_place(in_out.data(), local_count, recvcounts.data(), displs.data(), root, comm);
    });
    fvec.push_back([&](std::vector<T>& in_out) { gatherv_in_place(in_out, local_count, root, comm); });

    // perform gatherv_in_place and check the result
    for (auto fun : fvec) {
        std::vector<T> data = (comm.rank() == root) ? in_root : in_nonroot;
        fun(data);
        if (comm.rank() == root) {
            ASSERT_EQ(data, exp);
        }
    }
}

// Helper function for performing tests with varying counts per process.
template <typename T>
void perform_test(bool in_place = false) {
    simplemc::mpi::communicator comm {};
    const int size = comm.size();
    const int rank = comm.rank();
    const int count = rank + 1;

    // prepare input and expected output (each process sends (rank + 1) elements)
    std::vector<T> exp, in;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            exp.push_back(r * 10 + i);
            if constexpr (!std::is_arithmetic_v<T>) {
                exp.back().imag(-r * 10 - i);
            }
            in.push_back((r == rank ? exp.back() : T {}));
        }
    }

    // perform test for each rank as root
    for (int r = 0; r < size; ++r) {
        if (in_place) {
            // for in place gatherv:
            // - root needs full buffer with its data at correct position
            // - non-root needs only its local data
            const int disp = (rank * (rank + 1)) / 2;
            auto it = exp.begin() + disp;
            std::vector<T> local_data(it, it + count);
            check_gatherv_in_place<T>(in, local_data, count, exp, r);
        } else {
            const int disp = (rank * (rank + 1)) / 2;
            auto it = in.begin() + disp;
            check_gatherv(std::vector<T>(it, it + count), exp, r);
        }
    }
}

TEST(SimplemcMPI, GathervZeroValues) {
    simplemc::mpi::communicator comm {};
    std::vector<int> out_rg {};
    constexpr int root = 0;
    simplemc::mpi::gatherv(std::vector<int> {}, out_rg, root, comm);
    ASSERT_TRUE(out_rg.empty());
}

TEST(SimplemcMPI, GathervVaryingCounts) {
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

TEST(SimplemcMPI, GathervInPlaceVaryingCounts) {
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

TEST(SimplemcMPI, GathervUniformCounts) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    const int count = 3;

    // each process sends the same number of elements
    std::vector<int> in, exp;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < count; ++i) {
            exp.push_back(r * 100 + i);
            if (r == rank) {
                in.push_back(exp.back());
            }
        }
    }

    // gatherv and check result for each root
    for (int r = 0; r < size; ++r) {
        std::vector<int> result(exp.size());
        simplemc::mpi::gatherv(in, result, r, comm);
        if (comm.rank() == r) {
            ASSERT_EQ(result, exp);
        }
    }
}

TEST(SimplemcMPI, GathervWithSplitCommunicator) {
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

    // each process sends (rank + 1) elements
    std::vector<int> in, exp;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            exp.push_back(r * 100 + i);
            if (r == rank) {
                in.push_back(exp.back());
            }
        }
    }

    // gatherv and check result
    std::vector<int> result(exp.size());
    simplemc::mpi::gatherv(in, result, root, sub_comm);
    if (sub_comm.rank() == root) {
        ASSERT_EQ(result, exp);
    }
}

TEST(SimplemcMPI, GathervEmptyFromSomeProcesses) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    constexpr int root = 0;

    // only even ranks send data
    std::vector<double> in, exp;
    for (int r = 0; r < size; ++r) {
        if (r % 2 == 0) {
            for (int i = 0; i < 2; ++i) {
                exp.push_back(static_cast<double>(r * 10 + i));
                if (r == rank) {
                    in.push_back(exp.back());
                }
            }
        }
    }

    // gatherv and check result
    std::vector<double> result(exp.size());
    simplemc::mpi::gatherv(in, result, root, comm);
    if (comm.rank() == root) {
        ASSERT_EQ(result, exp);
    }
}

TEST(SimplemcMPI, GathervStrings) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    if (size > 10) {
        GTEST_SKIP() << "Test works only for <= 10 processes";
    }

    // each process sends a string with length equal to (rank + 1)
    std::string expected {}, in_str {};
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            expected += fmt::format("{}", r);
            if (r == rank) {
                in_str += fmt::format("{}", r);
            }
        }
    }

    // gatherv and check result for each root
    for (int r = 0; r < size; ++r) {
        std::string result(expected.size(), ' ');
        simplemc::mpi::gatherv(in_str, result, r, comm);
        if (comm.rank() == r) {
            ASSERT_EQ(result, expected);
        }
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
