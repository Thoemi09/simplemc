#include "../gtest-mpi-listener.hpp"
#include "./mc_test_utils.hpp"

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/mpi.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>

using namespace simplemc;

namespace {

// User measurement wrapping a library accumulator. Provides an ADL `simplemc_mpi_collect`
// overload that reduces the inner accumulator via its own (library-shipped) `simplemc_mpi_collect`.
struct mean_meas {
    using acc_t = mean_acc<double>;
    acc_t acc { 1 };
    void measure() {}
    friend void simplemc_mpi_collect(const mpi::communicator& comm, mean_meas& m) { simplemc_mpi_collect(comm, m.acc); }
};

// Read back everything written to a temporary file.
std::string read_all(std::FILE* fp) {
    std::rewind(fp); // NOLINT
    std::string content;
    for (int c = std::fgetc(fp); c != EOF; c = std::fgetc(fp)) {
        content.push_back(static_cast<char>(c));
    }
    return content;
}

// Pretend to perform the dummy update so its run counters read (nprops, naccs, nimps).
void drive_counters(update<dummy_update>& u, std::uint64_t nprops, std::uint64_t naccs, std::uint64_t nimps) {
    for (std::uint64_t i = 0; i < nprops; ++i) {
        u.attempt();
    }
    for (std::uint64_t i = 0; i < naccs; ++i) {
        u.accept();
    }
    for (std::uint64_t i = 0; i < nimps; ++i) {
        u.mark_impossible();
    }
}

} // namespace

// Test fixture for MC MPI tests.
class SimplemcMCMPI : public ::testing::Test {
protected:
    void SetUp() override {
        rank = comm.rank();
        size = comm.size();
    }
    mpi::communicator comm;
    int rank { 0 };
    int size { 0 };
};

// Test collecting simulation statistics.
TEST_F(SimplemcMCMPI, SimulationStatsAllReduceSumsCounters) {
    // prepare rank specific stats
    simulation_stats st;
    st.cumulative_steps = static_cast<std::uint64_t>(100) * (rank + 1);
    st.cumulative_time = 0.25 * (rank + 1);

    // collect across ranks
    simplemc_mpi_collect(comm, st);

    // check the collected values
    const auto expected_int_sum = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(st.cumulative_steps, 100u * expected_int_sum);
    EXPECT_DOUBLE_EQ(st.cumulative_time, 0.25 * static_cast<double>(size * (size + 1)) / 2.0);
}

// Test collecting a single update.
TEST_F(SimplemcMCMPI, UpdateAllReducesCountersAndDelegatesToWrapper) {
    // prepare rank-specific update counters
    const auto r = static_cast<std::uint64_t>(rank);
    update u { dummy_update {}, "u", 1.0 };
    drive_counters(u, 20 * (r + 1), 5 * (r + 1), 2 * (r + 1));
    drive_counters(u, 10 * (r + 1), 3 * (r + 1), (rank == 0) ? 1 : 0);

    // collect across ranks
    simplemc_mpi_collect(comm, u);

    // check the collected values
    const auto s = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(u.stats().nprops, 30u * s);
    EXPECT_EQ(u.stats().naccs, 8u * s);
    EXPECT_EQ(u.stats().nimps, 2u * s + 1u);

    // name / weight are local registration data and must be untouched
    EXPECT_EQ(u.name(), "u");
    EXPECT_DOUBLE_EQ(u.stats().weight, 1.0);
}

// Test collecting an update set.
TEST_F(SimplemcMCMPI, UpdateSetReducesEveryEntry) {
    // prepare rank-specific update counters
    update_set us { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 2.0 } };
    const auto r = static_cast<std::uint64_t>(rank);
    drive_counters(us.get<0>(), r + 1, 0, 0);
    drive_counters(us.get<1>(), 2 * (r + 1), 0, 0);

    // collect across ranks
    simplemc_mpi_collect(comm, us);

    // check the collected values
    const auto s = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(us.get<0>().stats().nprops, s);
    EXPECT_EQ(us.get<1>().stats().nprops, 2u * s);
}

// Test collecting a measurement set.
TEST_F(SimplemcMCMPI, MeasurementSetForwardsToWrapperHook) {
    // prepare rank-specific measurements
    mean_meas mm;
    mm.acc << static_cast<double>(rank + 1);
    measurement_set ms { measurement { mm, "mean", true }, measurement { dummy_measurement {}, "counter", true } };

    // collect across ranks
    simplemc_mpi_collect(comm, ms);

    // check the collected values -- mean accumulator should now hold the global mean on every rank
    const auto* reduced = ms.get<mean_meas>("mean");
    ASSERT_NE(reduced, nullptr);
    mean_meas::acc_t ref { 1 };
    for (int r = 0; r < size; ++r) {
        ref << static_cast<double>(r + 1);
    }
    ASSERT_EQ(reduced->acc.count(), ref.count());
    EXPECT_NEAR(reduced->acc.mean(), ref.mean(), 1e-12);

    // dummy_measurement has no ADL `simplemc_mpi_collect` — silent no-op
    const auto* counter = ms.get<dummy_measurement>("counter");
    ASSERT_NE(counter, nullptr);
    EXPECT_EQ(counter->count, 0);
}

// Test collecting a composite of updates, measurements and simulation stats.
TEST_F(SimplemcMCMPI, CompositeReducesAllComponents) {
    // prepare rank-specific components
    update_set updates { update { dummy_update {}, "u", 1.0 } };
    mean_meas mm;
    mm.acc << static_cast<double>(rank + 1);
    measurement_set meas { measurement { mm, "mean", true } };
    simulation_stats stats;
    drive_counters(updates.get<0>(), static_cast<std::uint64_t>(rank) + 1, 0, 0);
    stats.cumulative_steps = static_cast<std::uint64_t>(rank) + 1;

    // collect across ranks
    simplemc_mpi_collect(comm, updates, meas, stats);

    // check the collected components
    const auto s = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(updates.get<0>().stats().nprops, s);
    EXPECT_EQ(stats.cumulative_steps, s);

    const auto* reduced = meas.get<mean_meas>("mean");
    ASSERT_NE(reduced, nullptr);
    mean_meas::acc_t ref { 1 };
    for (int r = 0; r < size; ++r) {
        ref << static_cast<double>(r + 1);
    }
    EXPECT_EQ(reduced->acc.count(), ref.count());
    EXPECT_NEAR(reduced->acc.mean(), ref.mean(), 1e-12);
}

// Test that the communicator print overloads emit on the root rank only.
TEST_F(SimplemcMCMPI, CommPrintOverloadsEmitOnRootOnly) {
    simulation_stats st;
    st.cumulative_steps = 10;
    st.cumulative_time = 1.0;
    const simulation_params params;
    update_set us { update { dummy_update {}, "u", 1.0 } };

    // default root (rank 0): every overload prints on the root rank and is silent elsewhere
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    print(comm, st, fp);
    print(comm, params, fp);
    print(comm, us.stats(), fp);
    print(comm, us.stats().front(), fp);
    EXPECT_EQ(read_all(fp).empty(), rank != 0);
    std::fclose(fp);

    // a non-zero root argument moves the emitting rank
    const int root = size - 1;
    std::FILE* fp2 = std::tmpfile();
    ASSERT_NE(fp2, nullptr);
    print(comm, st, fp2, root);
    EXPECT_EQ(read_all(fp2).empty(), rank != root);
    std::fclose(fp2);
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
