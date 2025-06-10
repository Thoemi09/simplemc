#include "./accs_fixture.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/var_acc.hpp>

#include <complex>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;
constexpr double tol = 1e-10;

} // namespace

// Check empty accumulators.
TEST_F(SimplemcAccs, VarAccEmpty) {
    using namespace simplemc;

    var_acc_single<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_empty(acc_sd);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    var_acc_single<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_empty(acc_sc);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    var_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_empty(acc_st_d);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    var_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_empty(acc_st_c);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    var_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_empty(acc_dyn_d);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    var_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_empty(acc_dyn_c);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Check variance accumulator of size 1.
TEST_F(SimplemcAccs, VarAccSingle) {
    // general set up
    using namespace simplemc;
    var_acc_single<double, standard> acc_std_d1, acc_std_d2;
    var_acc_single<double, welford> acc_wel_d1, acc_wel_d2;
    var_acc_single<std::complex<double>, standard> acc_std_c1, acc_std_c2;
    var_acc_single<std::complex<double>, welford> acc_wel_c1, acc_wel_c2;

    // fill accumulators
    const auto merge_size = steps / 2;
    for (int i = 0; i < merge_size; ++i) {
        acc_std_d1 << sp_d.samples[i](0);
        acc_wel_d1 << sp_d.samples[i](0);
        acc_std_c1 << sp_c.samples[i](0);
        acc_wel_c1 << sp_c.samples[i](0);
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_std_d2[0] << sp_d.samples[i](0);
        acc_wel_d2[0] << sp_d.samples[i](0);
        acc_std_c2[0] << sp_c.samples[i](0);
        acc_wel_c2[0] << sp_c.samples[i](0);
    }

    // check count
    ASSERT_EQ(acc_std_d1.count(), merge_size);
    ASSERT_EQ(acc_wel_d1.count(), merge_size);
    ASSERT_EQ(acc_std_c1.count(), merge_size);
    ASSERT_EQ(acc_wel_c1.count(), merge_size);
    ASSERT_EQ(acc_std_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_wel_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_std_c2.count(), steps - merge_size);
    ASSERT_EQ(acc_wel_c2.count(), steps - merge_size);

    // merge accumulators
    acc_std_d1 << acc_std_d2;
    acc_wel_d1 << acc_wel_d2;
    acc_std_c1 << acc_std_c2;
    acc_wel_c1 << acc_wel_c2;

    // check mean and variance
    const auto m_d = sample_mean(sp_d);
    const auto v_d = sample_variance(sp_d);
    const auto m_c = sample_mean(sp_c);
    const auto [r_c, i_c, ri_c] = sample_variance(sp_c);
    check_near(acc_std_d1.mean(), m_d[0], tol);
    check_near(acc_std_d1.variance_of_data(), v_d[0], tol);
    check_near(acc_wel_d1.mean(), m_d[0], tol);
    check_near(acc_wel_d1.variance_of_data(), v_d[0], tol);
    check_near(acc_std_c1.mean(), m_c[0], tol);
    check_near(acc_std_c1.variance_of_real_data(), r_c[0], tol);
    check_near(acc_std_c1.variance_of_imag_data(), i_c[0], tol);
    check_near(acc_std_c1.covariance_of_real_and_imag_data(), ri_c[0], tol);
    check_near(acc_wel_c1.mean(), m_c[0], tol);
    check_near(acc_wel_c1.variance_of_real_data(), r_c[0], tol);
    check_near(acc_wel_c1.variance_of_imag_data(), i_c[0], tol);
    check_near(acc_wel_c1.covariance_of_real_and_imag_data(), ri_c[0], tol);
}

// Check variance accumulator using the full random vectors.
TEST_F(SimplemcAccs, VarAccVector) {
    // general set up
    using namespace simplemc;
    var_acc_static<double, size, standard> acc_std_d1, acc_std_d2;
    var_acc_dynamic<double, welford> acc_wel_d1(size), acc_wel_d2(size);
    var_acc_static<std::complex<double>, size, standard> acc_std_c1, acc_std_c2;
    var_acc_dynamic<std::complex<double>, welford> acc_wel_c1(size), acc_wel_c2(size);

    // fill accumulators
    const auto merge_size = steps / 2;
    std::vector<long> idxs(3);
    std::iota(idxs.begin(), idxs.end(), 0l);
    for (int i = 0; i < merge_size; ++i) {
        acc_std_d1 << sp_d.samples[i].matrix();
        acc_wel_d1 << sp_d.samples[i].matrix();
        acc_std_c1 << sp_c.samples[i].matrix();
        acc_wel_c1 << sp_c.samples[i].matrix();
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_std_d2.accumulate(sp_d.samples[i], 0);
        acc_wel_d2.accumulate(sp_d.samples[i], idxs);
        acc_std_c2.accumulate(sp_c.samples[i], 0);
        acc_wel_c2.accumulate(sp_c.samples[i], idxs);
    }

    // check count
    ASSERT_EQ(acc_std_d1.count(), merge_size);
    ASSERT_EQ(acc_wel_d1.count(), merge_size);
    ASSERT_EQ(acc_std_c1.count(), merge_size);
    ASSERT_EQ(acc_wel_c1.count(), merge_size);
    ASSERT_EQ(acc_std_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_wel_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_std_c2.count(), steps - merge_size);
    ASSERT_EQ(acc_wel_c2.count(), steps - merge_size);

    // merge accumulators
    acc_std_d1 << acc_std_d2;
    acc_wel_d1 << acc_wel_d2;
    acc_std_c1 << acc_std_c2;
    acc_wel_c1 << acc_wel_c2;

    // check mean and variance
    const auto m_d = sample_mean(sp_d);
    const auto v_d = sample_variance(sp_d);
    const auto m_c = sample_mean(sp_c);
    const auto [r_c, i_c, ri_c] = sample_variance(sp_c);
    check_range_near(acc_std_d1.mean(), m_d, tol);
    check_range_near(acc_std_d1.variance_of_data(), v_d, tol);
    check_range_near(acc_wel_d1.mean(), m_d, tol);
    check_range_near(acc_wel_d1.variance_of_data(), v_d, tol);
    check_range_near(acc_std_c1.mean(), m_c, tol);
    check_range_near(acc_std_c1.variance_of_real_data(), r_c, tol);
    check_range_near(acc_std_c1.variance_of_imag_data(), i_c, tol);
    check_range_near(acc_std_c1.covariance_of_real_and_imag_data(), ri_c, tol);
    check_range_near(acc_wel_c1.mean(), m_c, tol);
    check_range_near(acc_wel_c1.variance_of_real_data(), r_c, tol);
    check_range_near(acc_wel_c1.variance_of_imag_data(), i_c, tol);
    check_range_near(acc_wel_c1.covariance_of_real_and_imag_data(), ri_c, tol);
}

// Check variance accumulator using only part of random vectors.
TEST_F(SimplemcAccs, VarAccIndividualIndices) {
    // general set up
    using namespace simplemc;
    var_acc_static<double, size, standard> acc_std_d1, acc_std_d2;
    var_acc_dynamic<double, welford> acc_wel_d1(size), acc_wel_d2(size);
    var_acc_static<std::complex<double>, size, standard> acc_std_c1, acc_std_c2;
    var_acc_dynamic<std::complex<double>, welford> acc_wel_c1(size), acc_wel_c2(size);

    // fill accumulators
    const auto merge_size = steps / 2;
    std::vector<long> idxs { 0, size - 1 };
    const auto idx = 1;
    for (int i = 0; i < merge_size; ++i) {
        acc_std_d1[idx] << sp_d.samples[i](idx);
        acc_wel_d1.accumulate(sp_d.samples[i](idxs), idxs);
        acc_std_c1[idx] << sp_c.samples[i](idx);
        acc_wel_c1.accumulate(sp_c.samples[i](idxs), idxs);
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_std_d2[idx] << sp_d.samples[i](idx);
        acc_std_c2[idx] << sp_c.samples[i](idx);
        auto mva_d = acc_wel_d2.make_mva();
        auto mva_c = acc_wel_c2.make_mva();
        for (auto j : idxs) {
            mva_d[j] << sp_d.samples[i](j);
            mva_c[j] << sp_c.samples[i](j);
        }
        mva_d.increment_count();
        mva_c.increment_count();
    }

    // check count
    ASSERT_EQ(acc_std_d1.count(), merge_size);
    ASSERT_EQ(acc_wel_d1.count(), merge_size);
    ASSERT_EQ(acc_std_c1.count(), merge_size);
    ASSERT_EQ(acc_wel_c1.count(), merge_size);
    ASSERT_EQ(acc_std_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_wel_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_std_c2.count(), steps - merge_size);
    ASSERT_EQ(acc_wel_c2.count(), steps - merge_size);

    // merge accumulators
    acc_std_d1 << acc_std_d2;
    acc_wel_d1 << acc_wel_d2;
    acc_std_c1 << acc_std_c2;
    acc_wel_c1 << acc_wel_c2;

    // check mean and variance
    auto m_std_d = sample_mean(sp_d);
    auto v_std_d = sample_variance(sp_d);
    auto m_std_c = sample_mean(sp_c);
    auto [r_std_c, i_std_c, ri_std_c] = sample_variance(sp_c);
    auto [m_wel_d, v_wel_d] = std::tuple(m_std_d, v_std_d);
    auto [m_wel_c, r_wel_c, i_wel_c, ri_wel_c] = std::tuple(m_std_c, r_std_c, i_std_c, ri_std_c);
    for (int i = 0; i < size; ++i) {
        if (i != idx) {
            m_std_d(i) = 0.0;
            v_std_d(i) = 0.0;
            m_std_c(i) = { 0.0, 0.0 };
            r_std_c(i) = 0.0;
            i_std_c(i) = 0.0;
            ri_std_c(i) = 0.0;
        }
    }
    check_range_near(acc_std_d1.mean(), m_std_d, tol);
    check_range_near(acc_std_d1.variance_of_data(), v_std_d, tol);
    check_range_near(acc_std_c1.mean(), m_std_c, tol);
    check_range_near(acc_std_c1.variance_of_real_data(), r_std_c, tol);
    check_range_near(acc_std_c1.variance_of_imag_data(), i_std_c, tol);
    check_range_near(acc_std_c1.covariance_of_real_and_imag_data(), ri_std_c, tol);
    for (int i = 1; i < size - 1; ++i) {
        m_wel_d(i) = 0.0;
        v_wel_d(i) = 0.0;
        m_wel_c(i) = { 0.0, 0.0 };
        r_wel_c(i) = 0.0;
        i_wel_c(i) = 0.0;
        ri_wel_c(i) = 0.0;
    }
    check_range_near(acc_wel_d1.mean(), m_wel_d, tol);
    check_range_near(acc_wel_d1.variance_of_data(), v_wel_d, tol);
    check_range_near(acc_wel_c1.mean(), m_wel_c, tol);
    check_range_near(acc_wel_c1.variance_of_real_data(), r_wel_c, tol);
    check_range_near(acc_wel_c1.variance_of_imag_data(), i_wel_c, tol);
    check_range_near(acc_wel_c1.covariance_of_real_and_imag_data(), ri_wel_c, tol);
}

// Check blocked variance accumulator.
TEST_F(SimplemcAccs, VarAccBlocked) {
    // general set up
    using namespace simplemc;
    using block_acc_type = block_acc<var_acc_static<double, size, welford>>;
    const auto eff_steps = 1000;
    const auto block_size = steps / eff_steps;
    block_acc_type acc_wel_d1(block_size), acc_wel_d2(block_size);

    // fill accumulators
    const auto merge_size = steps / 2;
    std::vector<long> idxs(3);
    std::iota(idxs.begin(), idxs.end(), 0l);
    for (int i = 0; i < merge_size; ++i) {
        acc_wel_d1 << sp_d.samples[i].matrix();
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_wel_d2.accumulate(sp_d.samples[i], idxs);
    }

    // check count
    ASSERT_EQ(acc_wel_d1.count(), eff_steps / 2);
    ASSERT_EQ(acc_wel_d2.count(), eff_steps / 2);
    ASSERT_EQ(acc_wel_d1.total_count(), merge_size);
    ASSERT_EQ(acc_wel_d2.total_count(), steps - merge_size);

    // merge accumulators
    acc_wel_d1 << acc_wel_d2;

    // check mean and variance
    auto bsp_d = block_samples(sp_d, block_size);
    const auto m_d = sample_mean(bsp_d);
    const auto v_d = sample_variance(bsp_d);
    check_range_near(acc_wel_d1.accumulator().mean(), m_d, tol);
    check_range_near(acc_wel_d1.accumulator().variance_of_data(), v_d, tol);
}

// Check autocorrelation wrapper of variance accumulator.
TEST_F(SimplemcAccs, VarAccAutocorrelation) {
    // general set up
    using namespace simplemc;
    autocorr_acc<var_acc_static<double, size, welford>> acc_vec, acc_acc1, acc_acc2;
    autocorr_acc<var_acc_single<double, welford>> acc_single(1, 2, 5);

    // fill accumulators
    std::vector<long> idxs(3);
    std::iota(idxs.begin(), idxs.end(), 0l);
    for (int i = 0; i < steps; ++i) {
        acc_vec << sp_d.samples[i];
        acc_acc1.accumulate(sp_d.samples[i], idxs);
        acc_acc2.accumulate(sp_d.samples[i], 0);
        acc_single << sp_d.samples[i](0);
    }

    // check block variance accumulators with increasing block sizes and autocorrelation times
    using simplemc::accs::tau;
    auto [bsp_d, btau_v, btau_c] = blocking_autocorr(sp_d);
    const auto max_level = acc_vec.find_level(2);
    auto check_level = [&bsp_d, &btau_v](const auto& acc, auto i) {
        const auto v_d = sample_variance(bsp_d[i]);
        const auto av0_d = acc.accumulators()[0].variance_of_data();
        const auto av_d = acc.accumulators()[i].variance_of_data();
        const auto blsize = acc.block_sizes()[i];
        if constexpr (std::decay_t<decltype(acc)>::static_size == 1) {
            check_near(acc.accumulators()[i].variance_of_data(), v_d(0), tol);
            check_near(tau(av0_d, av_d, blsize), btau_v[i](0), tol);
        } else {
            check_range_near(acc.accumulators()[i].variance_of_data(), v_d, tol);
            check_range_near(tau(av0_d, av_d, blsize), btau_v[i], tol);
        }
    };
    for (std::size_t i = 0; i < max_level; ++i) {
        check_level(acc_single, i);
        check_level(acc_vec, i);
        check_level(acc_acc1, i);
        check_level(acc_acc2, i);
    }
}
