#include "./accs_test_traits.hpp"
#include "../gtest-mpi-listener.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/mpi.hpp>

#include <cstddef>
#include <vector>

namespace {

using namespace simplemc;
using namespace test_detail;

} // namespace

// MPI fixture: provides the communicator and rank info.
template <typename Tag>
class SimplemcAccsMPI : public ::testing::Test {
protected:
    void SetUp() override {
        rank = comm.rank();
        size = comm.size();
    }

    simplemc::mpi::communicator comm;
    int rank { 0 };
    int size { 0 };
};
TYPED_TEST_SUITE(SimplemcAccsMPI, AllAccTypes);

// Mean accumulator MPI test.
// Each rank accumulates one sample.
TYPED_TEST(SimplemcAccsMPI, MeanAccCollectSingleSample) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = mean_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    acc << data[this->rank % N];

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (int r = 0; r < this->size; ++r) {
        ref << data[r % N];
    }
    check_acc_equal(res, ref);
}

// Mean accumulator MPI test.
// Each rank accumulates strided samples.
TYPED_TEST(SimplemcAccsMPI, MeanAccCollectStrided) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = mean_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (int i = this->rank; i < N; i += this->size) {
        acc << data[i];
    }

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (int i = 0; i < N; ++i) {
        ref << data[i];
    }
    check_acc_equal(res, ref);
}

// Mean accumulator MPI test.
// Only rank 0 accumulates all samples, others are empty.
TYPED_TEST(SimplemcAccsMPI, MeanAccCollectAsymmetric) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = mean_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    if (this->rank == 0) {
        for (const auto& x : data) {
            acc << x;
        }
    }

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (const auto& x : data) {
        ref << x;
    }
    check_acc_equal(res, ref);
}

// Variance accumulator MPI test.
// Each rank accumulates one sample.
TYPED_TEST(SimplemcAccsMPI, VarAccCollectSingleSample) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    acc << data[this->rank % N];

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (int r = 0; r < this->size; ++r) {
        ref << data[r % N];
    }
    check_acc_equal(res, ref);
}

// Variance accumulator MPI test.
// Each rank accumulates strided samples.
TYPED_TEST(SimplemcAccsMPI, VarAccCollectStrided) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = var_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (int i = this->rank; i < N; i += this->size) {
        acc << data[i];
    }

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (int i = 0; i < N; ++i) {
        ref << data[i];
    }
    check_acc_equal(res, ref);
}

// Covariance accumulator MPI test.
// Each rank accumulates one sample.
TYPED_TEST(SimplemcAccsMPI, CovarAccCollectSingleSample) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    acc << data[this->rank % N];

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (int r = 0; r < this->size; ++r) {
        ref << data[r % N];
    }
    check_acc_equal(res, ref);
}

// Covariance accumulator MPI test.
// Each rank accumulates strided samples.
TYPED_TEST(SimplemcAccsMPI, CovarAccCollectStrided) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = covar_acc<T, A>;

    auto data = make_data<T>();
    auto acc = make_empty<acc_t, T>();
    for (int i = this->rank; i < N; i += this->size) {
        acc << data[i];
    }

    auto res = simplemc_mpi_collect(this->comm, acc);

    auto ref = make_empty<acc_t, T>();
    for (int i = 0; i < N; ++i) {
        ref << data[i];
    }
    check_acc_equal(res, ref);
}

// Block accumulator MPI test.
// block_acc<var_acc> with strided data distribution.
TYPED_TEST(SimplemcAccsMPI, BlockVarAccCollect) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using inner_t = var_acc<T, A>;
    using acc_t = block_acc<inner_t>;

    auto data = make_data<T>();
    auto acc = make_empty_block<acc_t, T>(5);
    for (int i = this->rank; i < N; i += this->size) {
        acc << data[i];
    }

    auto res = simplemc_mpi_collect(this->comm, acc);

    // build reference: simulate per-rank blocking, then merge inner accumulators
    auto ref = make_empty<inner_t, T>();
    for (int r = 0; r < this->size; ++r) {
        auto local = make_empty_block<acc_t, T>(5);
        for (int i = r; i < N; i += this->size) {
            local << data[i];
        }
        ref << local.accumulator();
    }
    check_acc_equal(res.accumulator(), ref);
}

// Block accumulator MPI test.
// block_acc<covar_acc> with strided data distribution.
TYPED_TEST(SimplemcAccsMPI, BlockCovarAccCollect) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using inner_t = covar_acc<T, A>;
    using acc_t = block_acc<inner_t>;

    auto data = make_data<T>();
    auto acc = make_empty_block<acc_t, T>(5);
    for (int i = this->rank; i < N; i += this->size) {
        acc << data[i];
    }

    auto res = simplemc_mpi_collect(this->comm, acc);

    // build reference: simulate per-rank blocking, then merge inner accumulators
    auto ref = make_empty<inner_t, T>();
    for (int r = 0; r < this->size; ++r) {
        auto local = make_empty_block<acc_t, T>(5);
        for (int i = r; i < N; i += this->size) {
            local << data[i];
        }
        ref << local.accumulator();
    }
    check_acc_equal(res.accumulator(), ref);
}

// Batch accumulator MPI test.
// Each rank adds 4 rank-specific samples, collect and compare.
TYPED_TEST(SimplemcAccsMPI, BatchAccCollect) {
    using T = typename TypeParam::sample_t;
    constexpr auto A = TypeParam::alg;
    using acc_t = batch_acc<T, A>;
    using macc_t = typename acc_t::mean_acc_type;

    auto acc = make_empty_batch<acc_t, T>(2);
    for (int i = 0; i < 4; ++i) {
        acc << make_sample<T>(this->rank * 4 + i);
    }

    auto res = simplemc_mpi_collect(this->comm, acc, false);

    // build reference: simulate each rank's batch_acc and collect all batches
    std::vector<macc_t> ref_batches;
    for (int r = 0; r < this->size; ++r) {
        auto local = make_empty_batch<acc_t, T>(2);
        for (int i = 0; i < 4; ++i) {
            local << make_sample<T>(r * 4 + i);
        }
        for (const auto& b : local.batches()) {
            ref_batches.push_back(b);
        }
    }

    ASSERT_EQ(res.size(), ref_batches.size());
    for (std::size_t i = 0; i < res.size(); ++i) {
        check_acc_equal(res[i], ref_batches[i]);
    }
}

// Custom main for MPI.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    return RUN_ALL_TESTS();
}
