#include "../gtest-mpi-listener.hpp"

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/mpi.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

using namespace simplemc;

namespace {

// User measurement carrying no MPI-collectible state — used to verify the silent no-op path
// through basic_measurement::mpi_collect for types without an ADL `simplemc_mpi_collect` overload.
struct counter_meas {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

// User measurement wrapping a library accumulator. Provides an ADL `simplemc_mpi_collect`
// overload that reduces the inner accumulator via its own (library-shipped) `simplemc_mpi_collect`.
struct mean_meas {
    using acc_t = mean_acc<double>;
    acc_t acc { 1 };
    void measure() {}
    friend mean_meas simplemc_mpi_collect(const mpi::communicator& comm, const mean_meas& m) {
        return mean_meas { simplemc_mpi_collect(comm, m.acc) };
    }
};

// Minimal MC update for populating an update_set in tests.
struct null_update {
    double attempt() { return 1.0; }
    void accept() {}
};

} // namespace

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

TEST_F(SimplemcMCMPI, SimulationStatsAllReduceSumsCounters) {
    simulation_stats st;
    st.last_steps_done = static_cast<std::uint64_t>(rank + 1);              // ranks contribute 1..size
    st.last_runtime = 1.5 * (rank + 1);
    st.cumulative_steps = static_cast<std::uint64_t>(100 * (rank + 1));
    st.cumulative_time = 0.25 * (rank + 1);

    simplemc_mpi_collect(comm, st);

    const std::uint64_t expected_int_sum = static_cast<std::uint64_t>(size * (size + 1) / 2);
    const double expected_dbl_sum = 1.5 * static_cast<double>(size * (size + 1)) / 2.0;
    EXPECT_EQ(st.last_steps_done, expected_int_sum);
    EXPECT_DOUBLE_EQ(st.last_runtime, expected_dbl_sum);
    EXPECT_EQ(st.cumulative_steps, 100u * expected_int_sum);
    EXPECT_DOUBLE_EQ(st.cumulative_time, 0.25 * static_cast<double>(size * (size + 1)) / 2.0);
}

TEST_F(SimplemcMCMPI, UpdateAllReducesCountersAndDelegatesToWrapper) {
    update u { null_update {}, "u", 1.0 };
    u.nprops = 10 * (rank + 1);
    u.naccs = 3 * (rank + 1);
    u.nimps = (rank == 0) ? 1 : 0;
    u.cumulative_nprops = 1000 * (rank + 1);
    u.cumulative_naccs = 500 * (rank + 1);
    u.cumulative_nimps = 2 * (rank + 1);

    simplemc_mpi_collect(comm, u);

    const std::uint64_t s = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(u.nprops, 10u * s);
    EXPECT_EQ(u.naccs, 3u * s);
    EXPECT_EQ(u.nimps, 1u);
    EXPECT_EQ(u.cumulative_nprops, 1000u * s);
    EXPECT_EQ(u.cumulative_naccs, 500u * s);
    EXPECT_EQ(u.cumulative_nimps, 2u * s);

    // Name / weight are local registration data and must be untouched.
    EXPECT_EQ(u.name, "u");
    EXPECT_DOUBLE_EQ(u.weight, 1.0);
}

TEST_F(SimplemcMCMPI, UpdateSetReducesEveryEntry) {
    update_set us;
    us.add({ null_update {}, "a", 1.0 });
    us.add({ null_update {}, "b", 2.0 });
    us[0].nprops = static_cast<std::uint64_t>(rank + 1);
    us[1].nprops = static_cast<std::uint64_t>(2 * (rank + 1));

    simplemc_mpi_collect(comm, us);

    const std::uint64_t s = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(us[0].nprops, s);
    EXPECT_EQ(us[1].nprops, 2u * s);
}

TEST_F(SimplemcMCMPI, MeasurementSetForwardsToWrapperHook) {
    // Two measurements: one with an ADL simplemc_mpi_collect (mean_meas) and one without (counter_meas).
    measurement_set ms;
    mean_meas mm;
    mm.acc << static_cast<double>(rank + 1);
    ms.add({ mm, "mean", true });
    ms.add({ counter_meas {}, "counter", true });

    simplemc_mpi_collect(comm, ms);

    // The mean accumulator should now hold the global mean (1, 2, ..., size) on every rank.
    const auto* reduced = ms.get<mean_meas>("mean");
    ASSERT_NE(reduced, nullptr);
    mean_meas::acc_t ref { 1 };
    for (int r = 0; r < size; ++r) {
        ref << static_cast<double>(r + 1);
    }
    ASSERT_EQ(reduced->acc.count(), ref.count());
    EXPECT_NEAR(reduced->acc.mean(), ref.mean(), 1e-12);

    // counter_meas has no ADL `simplemc_mpi_collect` — silent no-op. The local pointer should be unchanged.
    EXPECT_NE(ms.get<counter_meas>("counter"), nullptr);
}

TEST_F(SimplemcMCMPI, CompositeReducesAllComponents) {
    update_set updates;
    updates.add({ null_update {}, "u", 1.0 });

    measurement_set meas;
    mean_meas mm;
    mm.acc << static_cast<double>(rank + 1);
    meas.add({ mm, "mean", true });

    simulation_stats stats;

    // Seed per-update and per-stats counters with rank-distinct values without running the kernel,
    // so the verification is independent of RNG behavior.
    updates[0].nprops = static_cast<std::uint64_t>(rank + 1);
    stats.last_steps_done = static_cast<std::uint64_t>(rank + 1);

    simplemc_mpi_collect(comm, updates, meas, stats);

    const std::uint64_t s = static_cast<std::uint64_t>(size * (size + 1) / 2);
    EXPECT_EQ(updates[0].nprops, s);
    EXPECT_EQ(stats.last_steps_done, s);

    const auto* reduced = meas.get<mean_meas>("mean");
    ASSERT_NE(reduced, nullptr);
    mean_meas::acc_t ref { 1 };
    for (int r = 0; r < size; ++r) {
        ref << static_cast<double>(r + 1);
    }
    EXPECT_EQ(reduced->acc.count(), ref.count());
    EXPECT_NEAR(reduced->acc.mean(), ref.mean(), 1e-12);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));
    return RUN_ALL_TESTS();
}
