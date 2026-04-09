#include "./accs_fixture_advanced.hpp"
#include "../gtest-mpi-listener.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/mpi.hpp>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;
constexpr double tol = 1e-7;

} // namespace

// Fixture class for testing the MPI interface of the simplemc-accs library.
class SimplemcAccsMPIAdvanced : public SimplemcAccsAdvanced {
protected:
    void SetUp() override {
        SimplemcAccsAdvanced::SetUp();
        const auto rank = comm.rank();
        nsamples = steps / comm.size();
        for (int i = 0; i < steps; ++i) {
            exp_macc_std_d << sp_d.samples[i];
            exp_macc_wel_d << sp_d.samples[i];
            exp_macc_std_c << sp_c.samples[i];
            exp_macc_wel_c << sp_c.samples[i];

            exp_vacc_std_d << sp_d.samples[i];
            exp_vacc_wel_d << sp_d.samples[i];
            exp_vacc_std_c << sp_c.samples[i];
            exp_vacc_wel_c << sp_c.samples[i];

            exp_cacc_std_d << sp_d.samples[i];
            exp_cacc_wel_d << sp_d.samples[i];
            exp_cacc_std_c << sp_c.samples[i];
            exp_cacc_wel_c << sp_c.samples[i];

            exp_blacc_wel_d << sp_d.samples[i];

            if (i >= nsamples * rank && i < nsamples * (rank + 1)) {
                macc_std_d << sp_d.samples[i];
                macc_wel_d << sp_d.samples[i];
                macc_std_c << sp_c.samples[i];
                macc_wel_c << sp_c.samples[i];

                vacc_std_d << sp_d.samples[i];
                vacc_wel_d << sp_d.samples[i];
                vacc_std_c << sp_c.samples[i];
                vacc_wel_c << sp_c.samples[i];

                cacc_std_d << sp_d.samples[i];
                cacc_wel_d << sp_d.samples[i];
                cacc_std_c << sp_c.samples[i];
                cacc_wel_c << sp_c.samples[i];

                blacc_wel_d << sp_d.samples[i];
            }
        }
    }

    int nsamples { 0 };
    simplemc::mpi::communicator comm;

    // mean accumulators
    simplemc::mean_acc_static<double, size, standard> macc_std_d, exp_macc_std_d;
    simplemc::mean_acc_static<double, size, welford> macc_wel_d, exp_macc_wel_d;
    simplemc::mean_acc_static<std::complex<double>, size, standard> macc_std_c, exp_macc_std_c;
    simplemc::mean_acc_static<std::complex<double>, size, welford> macc_wel_c, exp_macc_wel_c;

    // variance accumulators
    simplemc::var_acc_static<double, size, standard> vacc_std_d, exp_vacc_std_d;
    simplemc::var_acc_static<double, size, welford> vacc_wel_d, exp_vacc_wel_d;
    simplemc::var_acc_static<std::complex<double>, size, standard> vacc_std_c, exp_vacc_std_c;
    simplemc::var_acc_static<std::complex<double>, size, welford> vacc_wel_c, exp_vacc_wel_c;

    // covariance accumulators
    simplemc::covar_acc_static<double, size, standard> cacc_std_d, exp_cacc_std_d;
    simplemc::covar_acc_static<double, size, welford> cacc_wel_d, exp_cacc_wel_d;
    simplemc::covar_acc_static<std::complex<double>, size, standard> cacc_std_c, exp_cacc_std_c;
    simplemc::covar_acc_static<std::complex<double>, size, welford> cacc_wel_c, exp_cacc_wel_c;

    // block accumulator
    simplemc::block_acc<simplemc::var_acc_static<double, size, welford>> blacc_wel_d { 5 }, exp_blacc_wel_d { 5 };
};

// Test MPI routines for mean_acc.
TEST_F(SimplemcAccsMPIAdvanced, MeanAccumulator) {
    EXPECT_EQ(macc_std_d.count(), nsamples);
    EXPECT_EQ(macc_wel_d.count(), nsamples);
    EXPECT_EQ(macc_std_c.count(), nsamples);
    EXPECT_EQ(macc_wel_c.count(), nsamples);
    EXPECT_EQ(exp_macc_std_d.count(), steps);
    EXPECT_EQ(exp_macc_wel_d.count(), steps);
    EXPECT_EQ(exp_macc_std_c.count(), steps);
    EXPECT_EQ(exp_macc_wel_c.count(), steps);

    // double standard
    auto res_macc_std_d = mpi_collect(comm, macc_std_d);
    EXPECT_EQ(res_macc_std_d.count(), steps);
    check_near(res_macc_std_d.mdata(), exp_macc_std_d.mdata(), tol);

    // double welford
    auto res_macc_wel_d = mpi_collect(comm, macc_wel_d);
    EXPECT_EQ(res_macc_wel_d.count(), steps);
    check_near(res_macc_wel_d.mdata(), exp_macc_wel_d.mdata(), tol);

    // complex standard
    auto res_macc_std_c = mpi_collect(comm, macc_std_c);
    EXPECT_EQ(res_macc_std_c.count(), steps);
    check_near(res_macc_std_c.mdata(), exp_macc_std_c.mdata(), tol);

    // complex welford
    auto res_macc_wel_c = mpi_collect(comm, macc_wel_c);
    EXPECT_EQ(res_macc_wel_c.count(), steps);
    check_near(res_macc_wel_c.mdata(), exp_macc_wel_c.mdata(), tol);
}

// Test MPI routines for var_acc.
TEST_F(SimplemcAccsMPIAdvanced, VarianceAccumulator) {
    EXPECT_EQ(vacc_std_d.count(), nsamples);
    EXPECT_EQ(vacc_wel_d.count(), nsamples);
    EXPECT_EQ(vacc_std_c.count(), nsamples);
    EXPECT_EQ(vacc_wel_c.count(), nsamples);
    EXPECT_EQ(exp_vacc_std_d.count(), steps);
    EXPECT_EQ(exp_vacc_wel_d.count(), steps);
    EXPECT_EQ(exp_vacc_std_c.count(), steps);
    EXPECT_EQ(exp_vacc_wel_c.count(), steps);

    // double standard
    auto res_vacc_std_d = mpi_collect(comm, vacc_std_d);
    EXPECT_EQ(res_vacc_std_d.count(), steps);
    check_near(res_vacc_std_d.mdata(), exp_vacc_std_d.mdata(), tol);
    check_near(res_vacc_std_d.cdata(), exp_vacc_std_d.cdata(), tol);

    // double welford
    auto res_vacc_wel_d = mpi_collect(comm, vacc_wel_d);
    EXPECT_EQ(res_vacc_wel_d.count(), steps);
    check_near(res_vacc_wel_d.mdata(), exp_vacc_wel_d.mdata(), tol);
    check_near(res_vacc_wel_d.cdata(), exp_vacc_wel_d.cdata(), tol);

    // complex standard
    auto res_vacc_std_c = mpi_collect(comm, vacc_std_c);
    EXPECT_EQ(res_vacc_std_c.count(), steps);
    check_near(res_vacc_std_c.mdata(), exp_vacc_std_c.mdata(), tol);
    check_near(res_vacc_std_c.rdata(), exp_vacc_std_c.rdata(), tol);
    check_near(res_vacc_std_c.idata(), exp_vacc_std_c.idata(), tol);
    check_near(res_vacc_std_c.cdata(), exp_vacc_std_c.cdata(), tol);

    // complex welford
    auto res_vacc_wel_c = mpi_collect(comm, vacc_wel_c);
    EXPECT_EQ(res_vacc_wel_c.count(), steps);
    check_near(res_vacc_wel_c.mdata(), exp_vacc_wel_c.mdata(), tol);
    check_near(res_vacc_wel_c.rdata(), exp_vacc_wel_c.rdata(), tol);
    check_near(res_vacc_wel_c.idata(), exp_vacc_wel_c.idata(), tol);
    check_near(res_vacc_wel_c.cdata(), exp_vacc_wel_c.cdata(), tol);
}

// Test MPI routines for covar_acc.
TEST_F(SimplemcAccsMPIAdvanced, CovarianceAccumulator) {
    using namespace simplemc;
    EXPECT_EQ(cacc_std_d.count(), nsamples);
    EXPECT_EQ(cacc_wel_d.count(), nsamples);
    EXPECT_EQ(cacc_std_c.count(), nsamples);
    EXPECT_EQ(cacc_wel_c.count(), nsamples);
    EXPECT_EQ(exp_cacc_std_d.count(), steps);
    EXPECT_EQ(exp_cacc_wel_d.count(), steps);
    EXPECT_EQ(exp_cacc_std_c.count(), steps);
    EXPECT_EQ(exp_cacc_wel_c.count(), steps);

    // double standard
    auto res_cacc_std_d = mpi_collect(comm, cacc_std_d);
    EXPECT_EQ(res_cacc_std_d.count(), steps);
    check_near(res_cacc_std_d.mdata(), exp_cacc_std_d.mdata(), tol);
    check_near(make_span(res_cacc_std_d.cdata()), make_span(exp_cacc_std_d.cdata()), tol);

    // double welford
    auto res_cacc_wel_d = mpi_collect(comm, cacc_wel_d);
    EXPECT_EQ(res_cacc_wel_d.count(), steps);
    check_near(res_cacc_wel_d.mdata(), exp_cacc_wel_d.mdata(), tol);
    check_near(make_span(res_cacc_wel_d.cdata()), make_span(exp_cacc_wel_d.cdata()), tol);

    // complex standard
    auto res_cacc_std_c = mpi_collect(comm, cacc_std_c);
    EXPECT_EQ(res_cacc_std_c.count(), steps);
    check_near(res_cacc_std_c.mdata(), exp_cacc_std_c.mdata(), tol);
    check_near(make_span(res_cacc_std_c.rdata()), make_span(exp_cacc_std_c.rdata()), tol);
    check_near(make_span(res_cacc_std_c.idata()), make_span(exp_cacc_std_c.idata()), tol);
    check_near(make_span(res_cacc_std_c.cdata()), make_span(exp_cacc_std_c.cdata()), tol);

    // complex welford
    auto res_cacc_wel_c = mpi_collect(comm, cacc_wel_c);
    EXPECT_EQ(res_cacc_wel_c.count(), steps);
    check_near(res_cacc_wel_c.mdata(), exp_cacc_wel_c.mdata(), tol);
    check_near(make_span(res_cacc_wel_c.rdata()), make_span(exp_cacc_wel_c.rdata()), tol);
    check_near(make_span(res_cacc_wel_c.idata()), make_span(exp_cacc_wel_c.idata()), tol);
    check_near(make_span(res_cacc_wel_c.cdata()), make_span(exp_cacc_wel_c.cdata()), tol);
}

// Test MPI routines for block_acc.
TEST_F(SimplemcAccsMPIAdvanced, BlockAccumulator) {
    EXPECT_EQ(blacc_wel_d.total_count(), nsamples);
    EXPECT_EQ(exp_blacc_wel_d.total_count(), steps);

    // double welford
    auto res_blacc_wel_d = mpi_collect(comm, blacc_wel_d);
    EXPECT_EQ(res_blacc_wel_d.total_count(), steps);
    check_near(res_blacc_wel_d.accumulator().mdata(), exp_blacc_wel_d.accumulator().mdata(), tol);
    check_near(res_blacc_wel_d.accumulator().cdata(), exp_blacc_wel_d.accumulator().cdata(), tol);
}

// Test MPI routines for batch_acc.
TEST_F(SimplemcAccsMPIAdvanced, BatchAccumulator) {
    const auto rank = comm.rank();
    const auto size = comm.size();
    const auto nbatches = 4;

    // 4 batches, each with 2^{rank + 1} samples
    simplemc::batch_acc<double> bacc { 1, nbatches };
    auto nsamples = std::vector<int>(size);
    for (int i = 0; i < size; ++i) {
        nsamples[i] = (1 << i) * nbatches * 2;
    }
    const auto max_count = (1 << (size - 1)) * 2;
    for (int i = 0; i < nsamples[rank]; ++i) {
        bacc << rank;
    }

    // gather all batches regardless of their count
    auto coll_diff_size = mpi_collect(comm, bacc, false);
    EXPECT_EQ(coll_diff_size.size(), size * nbatches);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < nbatches; ++j) {
            const auto idx = i * nbatches + j;
            EXPECT_DOUBLE_EQ(coll_diff_size[idx].mdata()[0], i);
            EXPECT_EQ(coll_diff_size[idx].count(), (1 << (i + 1)));
        }
    }

    // gather only batches with the same count
    auto coll_same_size = mpi_collect(comm, bacc, true);
    auto gathered_batches = 0;
    for (const auto& x : nsamples) {
        gathered_batches += x / max_count;
    }
    EXPECT_EQ(coll_same_size.size(), gathered_batches);
    int idx = 0;
    for (int i = 0; i < comm.size(); ++i) {
        const auto contributed = nsamples[i] / max_count;
        for (int j = 0; j < contributed; ++j) {
            EXPECT_DOUBLE_EQ(coll_same_size[idx].mdata()[0], i);
            EXPECT_EQ(coll_same_size[idx].count(), max_count);
            ++idx;
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
