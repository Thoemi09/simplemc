#include "./accs_fixture.hpp"
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
class SimplemcAccsMPI : public SimplemcAccs {
protected:
    void SetUp() override {
        SimplemcAccs::SetUp();
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
            if (i >= nsamples * rank && i < nsamples * (rank + 1)) {
                macc_std_d << sp_d.samples[i];
                macc_wel_d << sp_d.samples[i];
                macc_std_c << sp_c.samples[i];
                macc_wel_c << sp_c.samples[i];

                vacc_std_d << sp_d.samples[i];
                vacc_wel_d << sp_d.samples[i];
                vacc_std_c << sp_c.samples[i];
                vacc_wel_c << sp_c.samples[i];
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
};

// Test MPI routines for mean_acc.
TEST_F(SimplemcAccsMPI, MeanAccumulator) {
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
    check_range_near(res_macc_std_d.mdata(), exp_macc_std_d.mdata(), tol);

    // double welford
    auto res_macc_wel_d = mpi_collect(comm, macc_wel_d);
    EXPECT_EQ(res_macc_wel_d.count(), steps);
    check_range_near(res_macc_wel_d.mdata(), exp_macc_wel_d.mdata(), tol);

    // complex standard
    auto res_macc_std_c = mpi_collect(comm, macc_std_c);
    EXPECT_EQ(res_macc_std_c.count(), steps);
    check_range_near(res_macc_std_c.mdata(), exp_macc_std_c.mdata(), tol);

    // complex welford
    auto res_macc_wel_c = mpi_collect(comm, macc_wel_c);
    EXPECT_EQ(res_macc_wel_c.count(), steps);
    check_range_near(res_macc_wel_c.mdata(), exp_macc_wel_c.mdata(), tol);
}

// Test MPI routines for var_acc.
TEST_F(SimplemcAccsMPI, VarianceAccumulator) {
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
    check_range_near(res_vacc_std_d.mdata(), exp_vacc_std_d.mdata(), tol);
    check_range_near(res_vacc_std_d.vdata(), exp_vacc_std_d.vdata(), tol);

    // double welford
    auto res_vacc_wel_d = mpi_collect(comm, vacc_wel_d);
    EXPECT_EQ(res_vacc_wel_d.count(), steps);
    check_range_near(res_vacc_wel_d.mdata(), exp_vacc_wel_d.mdata(), tol);
    check_range_near(res_vacc_wel_d.vdata(), exp_vacc_wel_d.vdata(), tol);

    // complex standard
    auto res_vacc_std_c = mpi_collect(comm, vacc_std_c);
    EXPECT_EQ(res_vacc_std_c.count(), steps);
    check_range_near(res_vacc_std_c.mdata(), exp_vacc_std_c.mdata(), tol);
    check_range_near(res_vacc_std_c.rdata(), exp_vacc_std_c.rdata(), tol);
    check_range_near(res_vacc_std_c.idata(), exp_vacc_std_c.idata(), tol);
    check_range_near(res_vacc_std_c.cdata(), exp_vacc_std_c.cdata(), tol);

    // complex welford
    auto res_vacc_wel_c = mpi_collect(comm, vacc_wel_c);
    EXPECT_EQ(res_vacc_wel_c.count(), steps);
    check_range_near(res_vacc_wel_c.mdata(), exp_vacc_wel_c.mdata(), tol);
    check_range_near(res_vacc_wel_c.rdata(), exp_vacc_wel_c.rdata(), tol);
    check_range_near(res_vacc_wel_c.idata(), exp_vacc_wel_c.idata(), tol);
    check_range_near(res_vacc_wel_c.cdata(), exp_vacc_wel_c.cdata(), tol);
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
