#include "./accs_fixture.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/utils.hpp>

#include <complex>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;
constexpr double tol = 1e-10;

} // namespace

// Check empty accumulators.
TEST_F(SimplemcAccs, BatchAccEmpty) {
    using namespace simplemc;

    batch_acc_single<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_empty(acc_sd.mean_accumulator());
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    batch_acc_single<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_empty(acc_sc.mean_accumulator());
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    batch_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_empty(acc_st_d.mean_accumulator());
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    batch_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_empty(acc_st_c.mean_accumulator());
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    batch_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_empty(acc_dyn_d.mean_accumulator());
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    batch_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_empty(acc_dyn_c.mean_accumulator());
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Check batch accumulator of size 1.
TEST_F(SimplemcAccs, BatchAccSingle) {
    // general set up
    using namespace simplemc;
    std::vector<std::size_t> num_vec { 32, 64, 128 };
    std::vector<batch_acc_single<double, standard>> acc_vec;
    for (auto num : num_vec) {
        acc_vec.emplace_back(1, num);
    }
    autocorr_acc<var_acc_single<double, standard>> acc_autocorr;

    // fill accumulators
    for (int i = 0; i < steps; ++i) {
        for (auto& acc : acc_vec) {
            acc << sp_d.samples[i](0);
        }
        acc_autocorr << sp_d.samples[i](0);
    }

    // check accumulators with different batch numbers
    for (const auto& bacc : acc_vec) {
        EXPECT_EQ(bacc.count(), steps);
        const auto macc = bacc.mean_accumulator();
        check_near(macc.mean(), acc_autocorr.mean(), tol);
        const auto vacc = bacc.var_accumulator();
        const auto lvl = acc_autocorr.find_level(vacc.count());
        const auto& vacc_exp = acc_autocorr.accumulators()[lvl];
        EXPECT_EQ(vacc.count(), vacc_exp.count());
        check_range_near(vacc.mdata(), vacc_exp.mdata(), tol);
        check_range_near(vacc.vdata(), vacc_exp.vdata(), tol);
    }

    // check combining full batches
    auto vacc_32 = acc_vec[0].var_accumulator();
    auto vacc_64 = acc_vec[1].var_accumulator();
    auto vacc_64_c2 = acc_vec[1].var_accumulator(2);
    auto vacc_128_c2 = acc_vec[2].var_accumulator(2);
    auto vacc_128_c4 = acc_vec[2].var_accumulator(4);
    EXPECT_EQ(vacc_32.count(), vacc_64_c2.count());
    EXPECT_EQ(vacc_32.count(), vacc_128_c4.count());
    EXPECT_EQ(vacc_64.count(), vacc_128_c2.count());
    check_range_near(vacc_32.mdata(), vacc_64_c2.mdata(), tol);
    check_range_near(vacc_32.vdata(), vacc_64_c2.vdata(), tol);
    check_range_near(vacc_32.mdata(), vacc_128_c4.mdata(), tol);
    check_range_near(vacc_32.vdata(), vacc_128_c4.vdata(), tol);
    check_range_near(vacc_64.mdata(), vacc_128_c2.mdata(), tol);
    check_range_near(vacc_64.vdata(), vacc_128_c2.vdata(), tol);
}

// Check batch accumulator using the full random vectors.
TEST_F(SimplemcAccs, BatchAccVec) {
    // general set up
    using namespace simplemc;
    std::vector<std::size_t> num_vec { 32, 64, 128 };
    std::vector<batch_acc_dynamic<std::complex<double>, welford>> acc_vec;
    for (auto num : num_vec) {
        acc_vec.emplace_back(size, num);
    }
    autocorr_acc<covar_acc_dynamic<std::complex<double>, welford>> acc_autocorr(size);

    // fill accumulators
    for (int i = 0; i < steps / 2; ++i) {
        for (auto& acc : acc_vec) {
            acc << sp_c.samples[i];
        }
        acc_autocorr << sp_c.samples[i];
    }
    for (int i = steps / 2; i < steps; ++i) {
        for (auto& acc : acc_vec) {
            acc.accumulate(sp_c.samples[i], 0);
        }
        acc_autocorr << sp_c.samples[i];
    }

    // check accumulators with different batch numbers
    for (const auto& bacc : acc_vec) {
        EXPECT_EQ(bacc.count(), steps);
        const auto macc = bacc.mean_accumulator();
        check_range_near(macc.mean(), acc_autocorr.mean(), tol);
        const auto cacc = bacc.covar_accumulator();
        const auto lvl = acc_autocorr.find_level(cacc.count());
        const auto& cacc_exp = acc_autocorr.accumulators()[lvl];
        EXPECT_EQ(cacc.count(), cacc_exp.count());
        check_range_near(cacc.mdata(), cacc_exp.mdata(), tol);
        check_range_near(simplemc::make_span(cacc.rdata()), simplemc::make_span(cacc_exp.rdata()), tol);
        check_range_near(simplemc::make_span(cacc.idata()), simplemc::make_span(cacc_exp.idata()), tol);
    }
}

// Check batch accumulator using only part of random vectors.
TEST_F(SimplemcAccs, BatchAccIndividualIndices) {
    // general set up
    using namespace simplemc;
    std::vector<std::size_t> num_vec { 32, 64, 128 };
    std::vector<batch_acc_static<std::complex<double>, size, welford>> acc_vec;
    for (auto num : num_vec) {
        acc_vec.emplace_back(size, num);
    }
    autocorr_acc<covar_acc_static<std::complex<double>, size, welford>> acc_autocorr;

    // fill accumulators
    std::vector<long> idxs { 0, size - 1 };
    for (int i = 0; i < steps / 2; ++i) {
        for (auto& acc : acc_vec) {
            acc.accumulate(sp_c.samples[i](idxs), idxs);
        }
        acc_autocorr.accumulate(sp_c.samples[i](idxs), idxs);
    }
    for (int i = steps / 2; i < steps; ++i) {
        for (auto& acc : acc_vec) {
            auto mva = acc.make_mva();
            for (auto j : idxs) {
                mva[j] << sp_c.samples[i](j);
            }
            mva.increment_count();
            acc.check_and_advance();
        }
        acc_autocorr.accumulate(sp_c.samples[i](idxs), idxs);
    }

    // check accumulators with different batch numbers
    for (const auto& bacc : acc_vec) {
        EXPECT_EQ(bacc.count(), steps);
        const auto macc = bacc.mean_accumulator();
        check_range_near(macc.mean(), acc_autocorr.mean(), tol);
        const auto cacc = bacc.covar_accumulator();
        const auto lvl = acc_autocorr.find_level(cacc.count());
        const auto& cacc_exp = acc_autocorr.accumulators()[lvl];
        EXPECT_EQ(cacc.count(), cacc_exp.count());
        check_range_near(cacc.mdata(), cacc_exp.mdata(), tol);
        check_range_near(simplemc::make_span(cacc.rdata()), simplemc::make_span(cacc_exp.rdata()), tol);
        check_range_near(simplemc::make_span(cacc.idata()), simplemc::make_span(cacc_exp.idata()), tol);
    }
}
