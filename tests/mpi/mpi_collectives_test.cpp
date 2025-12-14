#include "../gtest-mpi-listener.hpp"
#include "../test_utils.hpp"

#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <complex>
#include <numbers>
#include <numeric>
#include <vector>

// Check the broadcasting of a single value.
template <typename T>
void check_single_broadcast(const simplemc::mpi::communicator& comm, const T& t, int root) {
    T value = (comm.rank() == root ? t : T {});
    simplemc::mpi::broadcast(comm, value, root);
    ASSERT_EQ(value, t);
}

// Check the broadcasting of a range.
template <typename R>
void check_range_broadcast(const simplemc::mpi::communicator& comm, const R& rg, int root) { // NOLINT
    R data {};
    if (comm.rank() == root) {
        data = rg;
    } else {
        data.resize(rg.size());
    }
    simplemc::mpi::broadcast(comm, data, root);
    ASSERT_EQ(data, rg);
}

// Test broadcasting of single values.
TEST(SimplemcMPI, BroadcastSingleValues) {
    using namespace std::complex_literals;
    using std::numbers::pi;
    simplemc::mpi::communicator comm;
    const int root = 0;
    check_single_broadcast(comm, static_cast<char>('x'), root);
    check_single_broadcast(comm, static_cast<signed char>('a'), root);
    check_single_broadcast(comm, static_cast<unsigned char>('B'), root);
    check_single_broadcast(comm, static_cast<short>(-100), root);
    check_single_broadcast(comm, static_cast<unsigned short>(100), root);
    check_single_broadcast(comm, static_cast<int>(-100), root);
    check_single_broadcast(comm, static_cast<unsigned int>(100), root);
    check_single_broadcast(comm, static_cast<long>(-12345), root);
    check_single_broadcast(comm, static_cast<unsigned long>(12345), root);
    check_single_broadcast(comm, static_cast<long long>(-12345678), root);
    check_single_broadcast(comm, static_cast<unsigned long long>(12345678), root);
    check_single_broadcast(comm, static_cast<float>(pi), root);
    check_single_broadcast(comm, static_cast<double>(pi), root);
    check_single_broadcast(comm, static_cast<bool>(false), root);
    check_single_broadcast(comm, static_cast<std::complex<double>>(1i), root);
}

// Test broadcasting of vectors.
TEST(SimplemcMPI, BroadcastVectors) {
    using namespace std::complex_literals;
    using std::numbers::pi;
    simplemc::mpi::communicator comm;
    const int root = 0;
    check_range_broadcast(comm, std::vector<char> { 'a' }, root);
    check_range_broadcast(comm, std::vector<signed char> { 'a', 'b' }, root);
    check_range_broadcast(comm, std::vector<unsigned char> { 'a', 'b', 'c' }, root);
    check_range_broadcast(comm, std::vector<short> { -1, 1, 20 }, root);
    check_range_broadcast(comm, std::vector<unsigned short> { 1, 20 }, root);
    check_range_broadcast(comm, std::vector<int> { -1 }, root);
    check_range_broadcast(comm, std::vector<unsigned int> { 1, 20 }, root);
    check_range_broadcast(comm, std::vector<long> { -1, 1, 20 }, root);
    check_range_broadcast(comm, std::vector<unsigned long> { 1, 20 }, root);
    check_range_broadcast(comm, std::vector<long long> { -1 }, root);
    check_range_broadcast(comm, std::vector<unsigned long long> { 200, 1, 20 }, root);
    check_range_broadcast(comm, std::vector<float> { pi, -pi }, root);
    check_range_broadcast(comm, std::vector<double> { pi, -pi }, root);
    check_range_broadcast(comm, std::vector<std::complex<double>> { 1i, -2.0 + 3i, 4 }, root);
    check_range_broadcast(comm, std::string { "Hello world!" }, root);
}

// Test gathering/scattering of/into single values.
TEST(SimplemcMPI, GatherSingleValuesAndScatterIntoSingleValues) {
    simplemc::mpi::communicator comm;
    const int root = 0;

    // allgather ranks in a vector
    std::vector<int> exp(comm.size()), res(comm.size());
    std::iota(exp.begin(), exp.end(), 0);
    simplemc::mpi::all_gather(comm, comm.rank(), res);
    ASSERT_EQ(exp, res);

    // gather ranks in a vector
    res = std::vector<int>((comm.rank() == root ? comm.size() : 0));
    simplemc::mpi::gather(comm, comm.rank(), res, root);
    if (comm.rank() == root) {
        ASSERT_EQ(exp, res);
    } else {
        ASSERT_TRUE(res.empty());
    }

    // scatter the gathered vector
    int res_scatter {};
    simplemc::mpi::scatter(comm, res, res_scatter, root);
    ASSERT_EQ(res_scatter, comm.rank());
}

// Test gathering/scattering of/into multiple vectors.
TEST(SimplemcMPI, GatherVectorsAndScatterIntoVectors) {
    using std::numbers::pi;
    simplemc::mpi::communicator comm;
    const int root = 0;
    const int rank = comm.rank();
    const long sz = 3;

    // assign values to a range
    auto assign = [&](auto it, auto rank) {
        for (int i = 1; i <= sz; ++i) {
            *it++ = pi * (rank + i);
        }
    };

    // prepare vector to be gathered
    std::vector<double> data(sz);
    assign(data.begin(), comm.rank());

    // prepare expected result
    std::vector<double> exp(comm.size() * sz);
    for (int i = 0; i < comm.size(); ++i) {
        assign(exp.begin() + i * sz, i);
    }

    // allgather
    std::vector<double> res(comm.size() * sz);
    simplemc::mpi::all_gather(comm, data, res);
    ASSERT_EQ(exp, res);

    // allgather in place
    res.assign(res.size(), 0);
    assign(res.begin() + rank * sz, rank);
    simplemc::mpi::all_gather_in_place(comm, res);
    ASSERT_EQ(exp, res);

    // gather
    res = std::vector<double>((comm.rank() == root ? comm.size() * sz : 0));
    simplemc::mpi::gather(comm, data, res, root);
    if (comm.rank() == root) {
        ASSERT_EQ(exp, res);
    } else {
        ASSERT_TRUE(res.empty());
    }

    // scatter
    std::vector<double> res_scatter(sz);
    simplemc::mpi::scatter(comm, res, res_scatter, root);
    ASSERT_EQ(data, res_scatter);

    // gather in place
    if (comm.rank() == root) {
        res.assign(res.size(), 0);
        assign(res.begin() + rank * sz, rank);
        simplemc::mpi::gather_in_place(comm, res, root);
        ASSERT_EQ(exp, res);
    } else {
        simplemc::mpi::gather_in_place(comm, data, root);
    }

    // scatter in place
    if (comm.rank() == root) {
        simplemc::mpi::scatter_in_place(comm, res, root);
        ASSERT_EQ(exp, res);
    } else {
        res_scatter.assign(sz, 0);
        simplemc::mpi::scatter_in_place(comm, res_scatter, root);
        ASSERT_EQ(data, res_scatter);
    }
}

// Test reducing single values.
TEST(SimplemcMPI, ReduceSingleValues) {
    simplemc::mpi::communicator comm;
    const int root = 0;
    const int rank = comm.rank();
    const int size = comm.size();
    const double sum = size * (size + 1) * 0.5;

    // value to reduce and expected result
    std::complex<double> value(rank + 1, -(rank + 1));
    std::complex<double> exp { sum, -sum };

    // allreduce
    std::complex<double> res {};
    simplemc::mpi::all_reduce(comm, value, res, MPI_SUM);
    check_near(res, exp);

    // allreduce in place
    res = value;
    simplemc::mpi::all_reduce_in_place(comm, res, MPI_SUM);
    check_near(res, exp);

    // reduce
    res = 0;
    simplemc::mpi::reduce(comm, value, res, MPI_SUM, root);
    if (rank == root) {
        check_near(res, exp);
    } else {
        check_near(res, std::complex<double> { 0, 0 });
    }

    // reduce in place
    res = value;
    simplemc::mpi::reduce_in_place(comm, res, MPI_SUM, root);
    if (rank == root) {
        check_near(res, exp);
    } else {
        check_near(res, value);
    }
}

// Test reducing vectors.
TEST(SimplemcMPI, ReduceVector) {
    simplemc::mpi::communicator comm;
    const int root = 0;
    const int rank = comm.rank();
    const int size = comm.size();
    const double sum = size * (size + 1) * 0.5;

    // vector to reduce and expected result
    auto value = std::complex<double>(rank + 1, -(rank + 1));
    std::vector<std::complex<double>> data { value, 2.0 * value };
    std::vector<std::complex<double>> exp { { sum, -sum }, { 2.0 * sum, -2.0 * sum } };

    // allreduce
    std::vector<std::complex<double>> res(data.size());
    simplemc::mpi::all_reduce(comm, data, res, MPI_SUM);
    check_range_near(res, exp);

    // allreduce in place
    res = data;
    simplemc::mpi::all_reduce_in_place(comm, res, MPI_SUM);
    check_range_near(res, exp);

    // reduce
    res.assign(2, 0.0);
    simplemc::mpi::reduce(comm, data, res, MPI_SUM, root);
    if (rank == root) {
        check_range_near(res, exp);
    } else {
        for (auto z : res) {
            check_near(z, std::complex<double> { 0, 0 });
        }
    }

    // reduce in place
    res = data;
    simplemc::mpi::reduce_in_place(comm, res, MPI_SUM, root);
    if (rank == root) {
        check_range_near(res, exp);
    } else {
        check_range_near(res, data);
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
