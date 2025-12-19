#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <fmt/format.h>
#include <simplemc/mpi.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

// Test all-gatherv with varying counts per process.
template <typename T>
void check_all_gatherv(const std::vector<T>& input, const std::vector<T>& expected) {
    using namespace simplemc::mpi;
    communicator comm {};
    std::vector<T> result(expected.size());
    all_gatherv(input, result, comm);
    ASSERT_EQ(result, expected);
}

// Test all-gatherv in place with varying counts per process.
template <typename T>
void check_all_gatherv_in_place(const std::vector<T>& input, int local_count, const std::vector<T>& expected) {
    using namespace simplemc::mpi;
    communicator comm {};
    std::vector<T> data = input;
    all_gatherv_in_place(data, local_count, comm);
    ASSERT_EQ(data, expected);
}

TEST(SimplemcMPI, AllGathervZeroValues) {
    simplemc::mpi::communicator comm {};
    std::vector<int> recvcounts(comm.size(), 0);
    std::vector<int> displs(comm.size(), 0);
    std::vector<int> in_values {};
    std::vector<int> out_values {};
    simplemc::mpi::all_gatherv(
        in_values.data(), 0, MPI_INT, out_values.data(), recvcounts.data(), displs.data(), MPI_INT, comm);
    ASSERT_TRUE(out_values.empty());
}

// Helper function for performing repeated tests with varying counts.
template <typename T>
void perform_varying_counts() {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();

    // each process sends (rank + 1) elements
    const int send_count = rank + 1;
    std::vector<T> in_values(send_count);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::vector<T> expected;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            if constexpr (std::is_arithmetic_v<T>) {
                expected.push_back(static_cast<T>(r * 100 + i));
            } else {
                expected.push_back(
                    T { static_cast<T::value_type>(r * 100 + i), static_cast<T::value_type>(-(r * 100 + i)) });
            }
            if (r == rank) {
                in_values[i] = expected.back();
            }
        }
        recvcounts[r] = r + 1;
        displs[r] = (r > 0) ? displs[r - 1] + recvcounts[r - 1] : 0;
    }

    // test all_gatherv
    check_all_gatherv<T>(in_values, expected);
}

TEST(SimplemcMPI, AllGathervVaryingCounts) {
    // signed integer types
    perform_varying_counts<short>();
    perform_varying_counts<int>();
    perform_varying_counts<long>();
    perform_varying_counts<long long>();

    // unsigned integer types
    perform_varying_counts<unsigned short>();
    perform_varying_counts<unsigned int>();
    perform_varying_counts<unsigned long>();
    perform_varying_counts<unsigned long long>();

    // floating point types
    perform_varying_counts<float>();
    perform_varying_counts<double>();
    perform_varying_counts<long double>();

    // complex types
    perform_varying_counts<std::complex<float>>();
    perform_varying_counts<std::complex<double>>();
    perform_varying_counts<std::complex<long double>>();
}

// Helper function for performing repeated in-place tests with varying counts.
template <typename T>
void perform_varying_counts_in_place() {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();

    // each process sends (rank + 1) elements
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::vector<T> expected, input;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            if constexpr (std::is_arithmetic_v<T>) {
                expected.push_back(static_cast<T>(r * 100 + i));
            } else {
                expected.push_back(
                    T { static_cast<T::value_type>(r * 100 + i), static_cast<T::value_type>(-(r * 100 + i)) });
            }
            input.push_back((r == rank ? expected.back() : T {}));
        }
        recvcounts[r] = r + 1;
        displs[r] = (r > 0) ? displs[r - 1] + recvcounts[r - 1] : 0;
    }

    // test all_gatherv_in_place
    check_all_gatherv_in_place<T>(input, recvcounts[rank], expected);
}

TEST(SimplemcMPI, AllGathervInPlaceVaryingCounts) {
    // signed integer types
    perform_varying_counts_in_place<short>();
    perform_varying_counts_in_place<int>();
    perform_varying_counts_in_place<long>();
    perform_varying_counts_in_place<long long>();

    // unsigned integer types
    perform_varying_counts_in_place<unsigned short>();
    perform_varying_counts_in_place<unsigned int>();
    perform_varying_counts_in_place<unsigned long>();
    perform_varying_counts_in_place<unsigned long long>();

    // floating point types
    perform_varying_counts_in_place<float>();
    perform_varying_counts_in_place<double>();
    perform_varying_counts_in_place<long double>();

    // complex types
    perform_varying_counts_in_place<std::complex<float>>();
    perform_varying_counts_in_place<std::complex<double>>();
    perform_varying_counts_in_place<std::complex<long double>>();
}

TEST(SimplemcMPI, AllGathervUniformCounts) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    const int count_per_process = 3;

    // each process sends the same number of elements
    std::vector<int> in_values(count_per_process);
    std::vector<int> recvcounts(size, count_per_process);
    std::vector<int> displs(size);
    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < count_per_process; ++i) {
            expected.push_back(r * 100 + i);
            if (r == rank) {
                in_values[i] = expected.back();
            }
        }
        displs[r] = r * count_per_process;
    }

    // test all_gatherv
    check_all_gatherv(in_values, expected);
}

TEST(SimplemcMPI, AllGathervWithSplitCommunicator) {
    simplemc::mpi::communicator comm {};
    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // split into even and odd ranks
    const int color = comm.rank() % 2;
    auto sub_comm = comm.split(color);

    const int rank = sub_comm.rank();
    const int size = sub_comm.size();

    // each process sends (rank + 1) elements
    const int send_count = rank + 1;
    std::vector<int> in_values(send_count);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            expected.push_back(r * 100 + i);
            if (r == rank) {
                in_values[i] = expected.back();
            }
        }
        recvcounts[r] = r + 1;
        displs[r] = (r > 0) ? displs[r - 1] + recvcounts[r - 1] : 0;
    }

    // all-gatherv within sub-communicator
    std::vector<int> result(expected.size());
    simplemc::mpi::all_gatherv(in_values, result, sub_comm);
    ASSERT_EQ(result, expected);
}

TEST(SimplemcMPI, AllGathervEmptyFromSomeProcesses) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    if (size < 2) {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    // only even ranks send data
    const int send_count = (rank % 2 == 0) ? 2 : 0;
    std::vector<double> in_values(send_count);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::vector<double> expected;
    for (int r = 0; r < size; ++r) {
        if (r % 2 == 0) {
            for (int i = 0; i < 2; ++i) {
                expected.push_back(static_cast<double>(r * 10 + i));
                if (r == rank) {
                    in_values[i] = expected.back();
                }
            }
        }
        recvcounts[r] = (r % 2 == 0) ? 2 : 0;
        displs[r] = (r > 0) ? displs[r - 1] + recvcounts[r - 1] : 0;
    }

    // test all_gatherv
    check_all_gatherv(in_values, expected);
}

TEST(SimplemcMPI, AllGathervStrings) {
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();
    const int size = comm.size();
    if (size > 10) {
        GTEST_SKIP() << "Test works only for <= 10 processes";
    }

    // each process sends a string with length equal to (rank + 1)
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::string expected {}, in_str {};
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < r + 1; ++i) {
            expected += fmt::format("{}", r);
            if (r == rank) {
                in_str += fmt::format("{}", r);
            }
        }
        recvcounts[r] = r + 1;
        displs[r] = (r > 0) ? displs[r - 1] + recvcounts[r - 1] : 0;
    }

    // test all_gatherv for strings
    std::string result(expected.size(), ' ');
    simplemc::mpi::all_gatherv(in_str, result, comm);
    ASSERT_EQ(result, expected);
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
