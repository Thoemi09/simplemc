#include "./accs_fixture_advanced.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/utils.hpp>

#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;
constexpr double tol = 1e-10;

} // namespace

// Check batch accumulator of size 1.
TEST_F(SimplemcAccsAdvanced, BatchAccSingle) {
    // general set up
    using namespace simplemc;
    std::vector<std::size_t> num_vec { 32, 64, 128 };
    std::vector<batch_acc<double, standard>> acc_vec;
    for (auto num : num_vec) {
        acc_vec.emplace_back(1, num);
    }
    autocorr_acc<var_acc<double, standard>> acc_autocorr;

    // fill accumulators
    for (int i = 0; i < steps; ++i) {
        for (auto& acc : acc_vec) {
            acc << sp_d.samples[i](0);
        }
        acc_autocorr << sp_d.samples[i](0);
    }

    // factory function
    auto dview = simplemc::ranges::transform_view(sp_d.samples, [](const auto& s) { return s(0); });
    auto acc_vec2 = make_batch_acc<standard>(dview, 256);
    acc_vec.push_back(std::move(acc_vec2));
    num_vec.push_back(256);

    // check accumulators with different batch numbers
    for (const auto& bacc : acc_vec) {
        EXPECT_EQ(bacc.count(), steps);
        const auto macc = bacc.mean_accumulator();
        check_near(macc.mean(), acc_autocorr.mean(), tol);
        const auto vacc = bacc.var_accumulator();
        const auto lvl = acc_autocorr.find_level(vacc.count());
        const auto& vacc_exp = acc_autocorr.accumulators()[lvl];
        EXPECT_EQ(vacc.count(), vacc_exp.count());
        check_near(vacc.mdata(), vacc_exp.mdata(), tol);
        check_near(vacc.cdata(), vacc_exp.cdata(), tol);
    }

    // check combining full batches
    auto vacc_32 = acc_vec[0].var_accumulator();
    auto vacc_64 = acc_vec[1].var_accumulator();
    auto vacc_64_c2 = acc_vec[1].var_accumulator(2);
    auto vacc_128 = acc_vec[2].var_accumulator();
    auto vacc_128_c2 = acc_vec[2].var_accumulator(2);
    auto vacc_128_c4 = acc_vec[2].var_accumulator(4);
    auto vacc_256_c2 = acc_vec[3].var_accumulator(2);
    auto vacc_256_c4 = acc_vec[3].var_accumulator(4);
    auto vacc_256_c8 = acc_vec[3].var_accumulator(8);
    EXPECT_EQ(vacc_32.count(), vacc_64_c2.count());
    EXPECT_EQ(vacc_32.count(), vacc_128_c4.count());
    EXPECT_EQ(vacc_64.count(), vacc_128_c2.count());
    EXPECT_EQ(vacc_32.count(), vacc_256_c8.count());
    EXPECT_EQ(vacc_64.count(), vacc_256_c4.count());
    EXPECT_EQ(vacc_128.count(), vacc_256_c2.count());
    check_near(vacc_32.mdata(), vacc_64_c2.mdata(), tol);
    check_near(vacc_32.cdata(), vacc_64_c2.cdata(), tol);
    check_near(vacc_32.mdata(), vacc_128_c4.mdata(), tol);
    check_near(vacc_32.cdata(), vacc_128_c4.cdata(), tol);
    check_near(vacc_32.mdata(), vacc_256_c8.mdata(), tol);
    check_near(vacc_32.cdata(), vacc_256_c8.cdata(), tol);
    check_near(vacc_64.mdata(), vacc_128_c2.mdata(), tol);
    check_near(vacc_64.cdata(), vacc_128_c2.cdata(), tol);
    check_near(vacc_64.mdata(), vacc_256_c4.mdata(), tol);
    check_near(vacc_64.cdata(), vacc_256_c4.cdata(), tol);
    check_near(vacc_128.mdata(), vacc_256_c2.mdata(), tol);
    check_near(vacc_128.cdata(), vacc_256_c2.cdata(), tol);

    // reset and verify empty
    acc_vec[0].reset();
    ASSERT_TRUE(acc_vec[0].empty());
    ASSERT_EQ(acc_vec[0].count(), 0UL);
}

// Check batch accumulator using the full random vectors.
TEST_F(SimplemcAccsAdvanced, BatchAccVec) {
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

    // factory function
    auto cview = simplemc::ranges::transform_view(sp_c.samples, [](const auto& s) { return Eigen::VectorXcd(s); });
    auto acc_vec2 = make_batch_acc(cview, 256);
    acc_vec.push_back(std::move(acc_vec2));
    num_vec.push_back(256);

    // check accumulators with different batch numbers
    for (const auto& bacc : acc_vec) {
        EXPECT_EQ(bacc.count(), steps);
        const auto macc = bacc.mean_accumulator();
        check_near(macc.mean(), acc_autocorr.mean(), tol);
        const auto cacc = bacc.covar_accumulator();
        const auto lvl = acc_autocorr.find_level(cacc.count());
        const auto& cacc_exp = acc_autocorr.accumulators()[lvl];
        EXPECT_EQ(cacc.count(), cacc_exp.count());
        check_near(cacc.mdata(), cacc_exp.mdata(), tol);
        check_near(simplemc::make_span(cacc.rdata()), simplemc::make_span(cacc_exp.rdata()), tol);
        check_near(simplemc::make_span(cacc.idata()), simplemc::make_span(cacc_exp.idata()), tol);
    }

    // reset and verify empty
    acc_vec[0].reset();
    ASSERT_TRUE(acc_vec[0].empty());
    ASSERT_EQ(acc_vec[0].count(), 0UL);
}

// Check batch accumulator using only part of random vectors.
TEST_F(SimplemcAccsAdvanced, BatchAccIndividualIndices) {
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
        check_near(macc.mean(), acc_autocorr.mean(), tol);
        const auto cacc = bacc.covar_accumulator();
        const auto lvl = acc_autocorr.find_level(cacc.count());
        const auto& cacc_exp = acc_autocorr.accumulators()[lvl];
        EXPECT_EQ(cacc.count(), cacc_exp.count());
        check_near(cacc.mdata(), cacc_exp.mdata(), tol);
        check_near(simplemc::make_span(cacc.rdata()), simplemc::make_span(cacc_exp.rdata()), tol);
        check_near(simplemc::make_span(cacc.idata()), simplemc::make_span(cacc_exp.idata()), tol);
    }
}
