#include "./accs_fixture.hpp"
#include "../gtest-mpi-listener.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/mpi.hpp>

namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;

} // namespace

// MPI fixture: distributes fixture data across ranks
// and provides the communicator.
class SimplemcAccsMPI : public SimplemcAccs {
protected:
    void SetUp() override {
        SimplemcAccs::SetUp();
        rank_ = comm_.rank();
        size_ = comm_.size();
    }

    simplemc::mpi::communicator comm_;
    int rank_ { 0 };
    int size_ { 0 };
};

// Test mpi_collect for mean_acc with static vectors.
TEST_F(SimplemcAccsMPI, MeanAccCollectVector) {
    // Each rank gets vec_d_data[rank % 4].
    mean_acc_static<double, 3> acc;
    acc << vec_d_data[rank_ % vec_d_n];

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    // With 4 ranks, each feeding one of the 4 vec_d_data
    // samples, the mean should converge to vec_d_mean.
    if (size_ == vec_d_n) {
        check_range_near(res.mean(), vec_d_mean, 1e-14);
    }
}

// Test mpi_collect for mean_acc with scalar double.
TEST_F(SimplemcAccsMPI, MeanAccCollectScalar) {
    // Each rank accumulates one scalar.
    mean_acc<double> acc;
    acc << static_cast<double>(rank_ + 1);

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    // Mean of {1, 2, ..., size} = (size+1)/2.
    const double expected_mean =
        (static_cast<double>(size_) + 1.0) / 2.0;
    check_near(res.mean(), expected_mean, 1e-14);
}

// Test mpi_collect for mean_acc with scalar complex.
TEST_F(SimplemcAccsMPI, MeanAccCollectComplex) {
    // Each rank gets cplx_5[rank % 5].
    mean_acc<cplx> acc;
    acc << cplx_5[rank_ % cplx_5_n];

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    if (size_ == cplx_5_n) {
        check_near(res.mean(), cplx_5_mean, 1e-14);
    }
}

// Test mpi_collect for var_acc with static vectors.
TEST_F(SimplemcAccsMPI, VarAccCollectVector) {
    // Each rank gets vec_d_data[rank % 4].
    var_acc_static<double, 3> acc;
    acc << vec_d_data[rank_ % vec_d_n];

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    if (size_ == vec_d_n) {
        check_range_near(res.mean(), vec_d_mean, 1e-14);
        check_range_near(
            res.variance_of_data(), vec_d_var, 1e-14);
    }
}

// Test mpi_collect for var_acc with scalar double.
TEST_F(SimplemcAccsMPI, VarAccCollectScalar) {
    // Each rank accumulates one scalar.
    var_acc<double> acc;
    acc << static_cast<double>(rank_ + 1);

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    // Mean of {1, 2, ..., size} = (size+1)/2.
    const double expected_mean =
        (static_cast<double>(size_) + 1.0) / 2.0;
    check_near(res.mean(), expected_mean, 1e-14);
}

// Test mpi_collect for var_acc with scalar complex.
TEST_F(SimplemcAccsMPI, VarAccCollectComplex) {
    // Each rank gets cplx_5[rank % 5].
    var_acc<cplx> acc;
    acc << cplx_5[rank_ % cplx_5_n];

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    if (size_ == cplx_5_n) {
        check_near(res.mean(), cplx_5_mean, 1e-14);
        check_near(res.variance_of_real_data(),
            cplx_5_var_re, 1e-14);
        check_near(res.variance_of_imag_data(),
            cplx_5_var_im, 1e-14);
    }
}

// Test mpi_collect for covar_acc with static vectors.
TEST_F(SimplemcAccsMPI, CovarAccCollectVector) {
    covar_acc_static<double, 3> acc;
    acc << vec_d_data[rank_ % vec_d_n];

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    if (size_ == vec_d_n) {
        check_range_near(res.mean(), vec_d_mean, 1e-14);
        // Check full covariance matrix.
        auto cov = res.covariance_of_data();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                check_near(cov(i, j), vec_d_cov(i, j), 1e-14);
            }
        }
    }
}

// Test mpi_collect for covar_acc with scalar complex.
TEST_F(SimplemcAccsMPI, CovarAccCollectComplex) {
    // Each rank gets cplx_5[rank % 5].
    covar_acc<cplx> acc;
    acc << cplx_5[rank_ % cplx_5_n];

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), static_cast<std::uint64_t>(size_));

    if (size_ == cplx_5_n) {
        check_near(res.mean(), cplx_5_mean, 1e-14);
    }
}

// Test mpi_collect for block_acc<var_acc>.
TEST_F(SimplemcAccsMPI, BlockVarAccCollect) {
    // Each rank accumulates 2 samples with block_size = 1
    // so each block is a single sample.
    block_acc<var_acc<double>> acc(1);

    // Rank 0: {1,2}, Rank 1: {3,4}, etc.
    for (int i = 0; i < 2; ++i) {
        const auto idx = rank_ * 2 + i;
        // Use modular indexing for safety with different
        // rank counts.
        acc << dbl_5[idx % dbl_5_n];
    }

    auto res = mpi_collect(comm_, acc);
    // After collecting, total_count should increase.
    ASSERT_GE(res.total_count(), acc.total_count());
}

// Test mpi_collect for block_acc<covar_acc>.
TEST_F(SimplemcAccsMPI, BlockCovarAccCollect) {
    // Each rank accumulates 2 samples with block_size = 1.
    block_acc<covar_acc<double>> acc(1);

    for (int i = 0; i < 2; ++i) {
        const auto idx = rank_ * 2 + i;
        acc << dbl_5[idx % dbl_5_n];
    }

    auto res = mpi_collect(comm_, acc);
    ASSERT_GE(res.total_count(), acc.total_count());
}

// Test mpi_collect for batch_acc.
TEST_F(SimplemcAccsMPI, BatchAccCollect) {
    // Each rank accumulates enough samples to produce full batches.
    batch_acc<double> acc(1, 2);
    for (int i = 0; i < 4; ++i) {
        acc << static_cast<double>(rank_ * 4 + i + 1);
    }

    auto res = mpi_collect(comm_, acc, false);
    // Each rank contributes its full batches.
    ASSERT_FALSE(res.empty());
}

// Test multiple samples per rank with merging.
TEST_F(SimplemcAccsMPI, MeanAccMultipleSamples) {
    // Rank 0: all 5 dbl_5 samples.
    // Other ranks: empty.
    mean_acc<double> acc;
    if (rank_ == 0) {
        for (const auto& x : dbl_5) { acc << x; }
    }

    auto res = mpi_collect(comm_, acc);
    ASSERT_EQ(res.count(), dbl_5_n);
    check_near(res.mean(), dbl_5_mean, 1e-14);
}

// Custom main for MPI.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(
        new GTestMPIListener::MPIEnvironment);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    ::testing::TestEventListener* l =
        listeners.Release(listeners.default_result_printer());

    listeners.Append(
        new GTestMPIListener::MPIWrapperPrinter(
            l, MPI_COMM_WORLD));

    return RUN_ALL_TESTS();
}
