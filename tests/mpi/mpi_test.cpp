/**
 * @file mpi_test.cpp
 * @brief Unit tests for the simplemc-mpi library.
 */

#include "../gtest-mpi-listener.hpp"

#include <simplemc/mpi.hpp>

#include <fmt/ranges.h>

#include <array>
#include <list>
#include <numeric>
#include <span>
#include <vector>

// Broadcast a single value or a range of values.
template <typename T>
void broadcast_check(const simplemc::mpi::communicator& comm, const T& t, int root) {
    T value;
    if (comm.rank() == root) {
        value = t;
    }
    if constexpr (simplemc::mpi::is_mpi_datatype_v<T>) {
        simplemc::mpi::broadcast(comm, value, root);
    } else {
        value.resize(t.size());
        simplemc::mpi::broadcast(comm, value, root);
    }
    ASSERT_EQ(value, t);
}

// Gather/Allgather a single value or a range of values.
template <typename T>
void gather_check(const simplemc::mpi::communicator& comm, const T& in, int root) {
    if constexpr (simplemc::mpi::is_mpi_datatype_v<T>) {
        std::vector<T> res, exp;
        exp.resize(comm.size(), in);
        res.resize(comm.size());
        simplemc::mpi::all_gather(comm, in, res);
        ASSERT_EQ(res, exp);
        res.assign(res.size(), T {});
        simplemc::mpi::gather(comm, in, res, root);
        if (comm.rank() == root) {
            ASSERT_EQ(res, exp);
        } else {
            ASSERT_EQ(res, std::vector<T>(res.size(), T {}));
        }
    } else {
        T res, exp;
        for (int i = 0; i < comm.size(); i++) {
            exp.insert(exp.end(), in.begin(), in.end());
        }
        res.resize(comm.size() * in.size());
        simplemc::mpi::all_gather(comm, in, res);
        ASSERT_EQ(res, exp);
        res.assign(res.size(), typename T::value_type());
        simplemc::mpi::gather(comm, in, res, root);
        if (comm.rank() == root) {
            ASSERT_EQ(res, exp);
        } else {
            ASSERT_EQ(res, T(res.size(), typename T::value_type()));
        }
    }
}

// Reduce/Allreduce a single value or a range of values.
template <typename T>
void reduce_check(const simplemc::mpi::communicator& comm, const T& in, const T& exp, MPI_Op op, int root) {
    if constexpr (simplemc::mpi::is_mpi_datatype_v<T>) {
        T res;
        simplemc::mpi::all_reduce(comm, in, res, op);
        ASSERT_EQ(res, exp);
        res = T {};
        simplemc::mpi::reduce(comm, in, res, op, root);
        if (comm.rank() == root) {
            ASSERT_EQ(res, exp);
        } else {
            ASSERT_EQ(res, T {});
        }
    } else {
        T res;
        res.resize(in.size());
        simplemc::mpi::all_reduce(comm, in, res, op);
        ASSERT_EQ(res, exp);
        res.assign(res.size(), typename T::value_type());
        simplemc::mpi::reduce(comm, in, res, op, root);
        if (comm.rank() == root) {
            ASSERT_EQ(res, exp);
        } else {
            ASSERT_EQ(res, T(res.size(), typename T::value_type()));
        }
    }
}

// Fixture class for testing the collective communications.
class SimplemcMPI : public ::testing::Test {
protected:
    [[nodiscard]] char char_val() const { return 'x'; }
    [[nodiscard]] signed char schar_val() const { return 'u'; }
    [[nodiscard]] unsigned char uchar_val() const { return 'P'; }
    [[nodiscard]] short short_val() const { return -1234; }
    [[nodiscard]] unsigned short ushort_val() const { return 1234; }
    [[nodiscard]] int int_val() const { return -123456; }
    [[nodiscard]] unsigned int uint_val() const { return 123456; }
    [[nodiscard]] long long_val() const { return -123456789012; }
    [[nodiscard]] unsigned long ulong_val() const { return 123456789012; }
    [[nodiscard]] long long llong_val() const { return -123456789012; }
    [[nodiscard]] unsigned long long ullong_val() const { return 123456789012; }
    [[nodiscard]] float float_val() const { return 12.3456789; }
    [[nodiscard]] double double_val() const { return 12.3456789012; }
    [[nodiscard]] bool bool_val() const { return true; }
    [[nodiscard]] std::complex<double> cdouble_val() const { return { 1.2345, 6.7890 }; }
    [[nodiscard]] std::string str_val() const { return "my test string"; }

    template <typename T>
    std::vector<T> make_vec(int size, T t) {
        return std::vector<T>(size, t);
    }

    simplemc::mpi::communicator comm;
};

// Custom type for testing MPI datatypes.
struct foo {
    int x;
};

// Test MPI with a hello world program.
TEST_F(SimplemcMPI, HelloWorldWithMPI) {
    fmt::print("Hello world, from {} of {} processes.\n", comm.rank(), comm.size());
}

// Test concepts and traits involving MPI datatypes.
TEST_F(SimplemcMPI, ConceptsAndTraits) {
    using darray_type = std::array<double, 10>;
    using farray_type = std::array<foo, 10>;
    ASSERT_TRUE(simplemc::mpi::mpi_range<std::vector<double>>);
    ASSERT_FALSE(simplemc::mpi::mpi_range<std::vector<foo>>);
    ASSERT_TRUE(simplemc::mpi::mpi_range<darray_type>);
    ASSERT_FALSE(simplemc::mpi::mpi_range<farray_type>);
    ASSERT_FALSE(simplemc::mpi::mpi_range<std::list<double>>);
    ASSERT_TRUE(simplemc::mpi::mpi_range<std::span<double>>);
    ASSERT_FALSE(simplemc::mpi::mpi_range<std::span<foo>>);
}

// Test broadcasting a single value.
TEST_F(SimplemcMPI, BroadcastSingleValues) {
    int root = 0;
    broadcast_check(comm, char_val(), root);
    broadcast_check(comm, schar_val(), root);
    broadcast_check(comm, uchar_val(), root);
    broadcast_check(comm, short_val(), root);
    broadcast_check(comm, ushort_val(), root);
    broadcast_check(comm, int_val(), root);
    broadcast_check(comm, uint_val(), root);
    broadcast_check(comm, long_val(), root);
    broadcast_check(comm, ulong_val(), root);
    broadcast_check(comm, llong_val(), root);
    broadcast_check(comm, ullong_val(), root);
    broadcast_check(comm, float_val(), root);
    broadcast_check(comm, double_val(), root);
    broadcast_check(comm, bool_val(), root);
    broadcast_check(comm, cdouble_val(), root);
}

// Test broadcasting multiple values.
TEST_F(SimplemcMPI, BroadcastVectors) {
    int root = 0;
    int size = 5;
    broadcast_check(comm, make_vec(size, char_val()), root);
    broadcast_check(comm, make_vec(size, schar_val()), root);
    broadcast_check(comm, make_vec(size, uchar_val()), root);
    broadcast_check(comm, make_vec(size, short_val()), root);
    broadcast_check(comm, make_vec(size, ushort_val()), root);
    broadcast_check(comm, make_vec(size, int_val()), root);
    broadcast_check(comm, make_vec(size, uint_val()), root);
    broadcast_check(comm, make_vec(size, long_val()), root);
    broadcast_check(comm, make_vec(size, ulong_val()), root);
    broadcast_check(comm, make_vec(size, llong_val()), root);
    broadcast_check(comm, make_vec(size, ullong_val()), root);
    broadcast_check(comm, make_vec(size, float_val()), root);
    broadcast_check(comm, make_vec(size, double_val()), root);
    broadcast_check(comm, make_vec(size, cdouble_val()), root);
    broadcast_check(comm, str_val(), root);
}

// Test gathering a single value.
TEST_F(SimplemcMPI, GatherSingleValues) {
    int root = 0;
    gather_check(comm, char_val(), root);
    gather_check(comm, schar_val(), root);
    gather_check(comm, uchar_val(), root);
    gather_check(comm, short_val(), root);
    gather_check(comm, ushort_val(), root);
    gather_check(comm, int_val(), root);
    gather_check(comm, uint_val(), root);
    gather_check(comm, long_val(), root);
    gather_check(comm, ulong_val(), root);
    gather_check(comm, llong_val(), root);
    gather_check(comm, ullong_val(), root);
    gather_check(comm, float_val(), root);
    gather_check(comm, double_val(), root);
    gather_check(comm, cdouble_val(), root);
}

// Test gathering multiple values.
TEST_F(SimplemcMPI, GatherVectors) {
    int root = 0;
    int size = 5;
    gather_check(comm, make_vec(size, char_val()), root);
    gather_check(comm, make_vec(size, schar_val()), root);
    gather_check(comm, make_vec(size, uchar_val()), root);
    gather_check(comm, make_vec(size, short_val()), root);
    gather_check(comm, make_vec(size, ushort_val()), root);
    gather_check(comm, make_vec(size, int_val()), root);
    gather_check(comm, make_vec(size, uint_val()), root);
    gather_check(comm, make_vec(size, long_val()), root);
    gather_check(comm, make_vec(size, ulong_val()), root);
    gather_check(comm, make_vec(size, llong_val()), root);
    gather_check(comm, make_vec(size, ullong_val()), root);
    gather_check(comm, make_vec(size, float_val()), root);
    gather_check(comm, make_vec(size, double_val()), root);
    gather_check(comm, make_vec(size, cdouble_val()), root);
    gather_check(comm, str_val(), root);
}

// Test reducing a single value.
TEST_F(SimplemcMPI, ReduceSingleValues) {
    int root = 0;
    int rank = comm.rank();
    int size = comm.size();
    int fac = size - 1;
    reduce_check<int>(comm, 5, 5 * size, MPI_SUM, root);
    reduce_check<double>(comm, 1.2, 1.2 * size, MPI_SUM, root);
    reduce_check<float>(comm, 2.3f * static_cast<float>(rank), 2.3f * static_cast<float>(fac), MPI_MAX, root);
    std::complex<double> cval { 1.0, 1.0 };
    auto cexp = cval * static_cast<double>(size);
    reduce_check<std::complex<double>>(comm, cval, cexp, MPI_SUM, root);
}

// Test reducing multiple values.
TEST_F(SimplemcMPI, ReduceVectors) {
    int root = 0;
    int rank = comm.rank();
    int size = comm.size();
    int fac = size - 1;
    reduce_check<std::vector<int>>(comm, { 3, 5 }, { 3 * size, 5 * size }, MPI_SUM, root);
    reduce_check<std::vector<double>>(comm, { 1.2, 2.3 }, { 1.2 * size, 2.3 * size }, MPI_SUM, root);
    reduce_check<std::vector<double>>(comm, { 1.2 * rank, 2.3 * rank }, { 1.2 * fac, 2.3 * fac }, MPI_MAX, root);
}

// Test comparing a value across processes.
TEST_F(SimplemcMPI, AllEqual) {
    int val = 1000;
    ASSERT_TRUE(simplemc::mpi::all_equal(comm, val));
    if (comm.size() > 1) {
        ASSERT_FALSE(simplemc::mpi::all_equal(comm, comm.rank()));
    } else {
        ASSERT_TRUE(simplemc::mpi::all_equal(comm, comm.rank()));
    }
}

// Test scattering vectors.
TEST_F(SimplemcMPI, ScatterVectors) {
    int root = 0;
    int num_el = 3;
    int rank = comm.rank();
    std::vector<int> vec(comm.size());
    if (rank == root) {
        std::iota(vec.begin(), vec.end(), 0);
    }
    int res {};
    simplemc::mpi::scatter(comm, vec, res, root);
    ASSERT_EQ(res, comm.rank());
    if (rank == root) {
        vec.resize(static_cast<std::size_t>(comm.size()) * num_el);
        std::iota(vec.begin(), vec.end(), 0);
    }
    std::vector<int> res_vec(num_el);
    simplemc::mpi::scatter(comm, vec, res_vec, root);
    std::vector<int> exp { rank * 3, rank * 3 + 1, rank * 3 + 2 };
    ASSERT_EQ(res_vec, exp);
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
