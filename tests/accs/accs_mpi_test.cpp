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
            exp_macc_d << sp_d.samples[i];
            exp_macc_c << sp_c.samples[i];
            if (i >= nsamples * rank && i < nsamples * (rank + 1)) {
                macc_d << sp_d.samples[i];
                macc_c << sp_c.samples[i];
            }
        }
    }
    int nsamples { 0 };
    simplemc::mean_acc_static<double, size, standard> macc_d;
    simplemc::mean_acc_static<double, size, standard> exp_macc_d;
    simplemc::mean_acc_static<std::complex<double>, size, welford> macc_c;
    simplemc::mean_acc_static<std::complex<double>, size, welford> exp_macc_c;
    simplemc::mpi::communicator comm;
};

TEST_F(SimplemcAccsMPI, MeanAccumulator) {
    EXPECT_EQ(macc_d.count(), nsamples);
    EXPECT_EQ(macc_c.count(), nsamples);
    EXPECT_EQ(exp_macc_d.count(), steps);
    EXPECT_EQ(exp_macc_c.count(), steps);
    auto res_macc_d = mpi_collect(comm, macc_d);
    EXPECT_EQ(res_macc_d.count(), steps);
    check_range_near(res_macc_d.mdata(), exp_macc_d.mdata(), tol);
    auto res_macc_c = mpi_collect(comm, macc_c);
    EXPECT_EQ(res_macc_c.count(), steps);
    check_range_near(res_macc_c.mdata(), exp_macc_c.mdata(), tol);
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
