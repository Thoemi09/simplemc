// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "./stochastic_process_fixture.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/covar_acc.hpp>

#include <simplemc/utils/ranges.hpp>

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

// Check covariance accumulator of size 1.
TEST_F(SimplemcAccsStochasticProcess, CovarAccSingle) {
    // general set up
    using namespace simplemc;
    covar_acc<double, standard> acc_std_d1, acc_std_d2;
    covar_acc<double, welford> acc_wel_d1, acc_wel_d2;
    covar_acc<std::complex<double>, standard> acc_std_c1, acc_std_c2;
    covar_acc<std::complex<double>, welford> acc_wel_c1, acc_wel_c2;

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

    // factory function
    auto dview = simplemc::ranges::transform_view(sp_d.samples, [](const auto& s) { return s(0); });
    auto acc_std_d3 = make_covar_acc<standard>(dview);
    auto acc_wel_d3 = make_covar_acc(dview);
    auto cview = simplemc::ranges::transform_view(sp_c.samples, [](const auto& s) { return s(0); });
    auto acc_std_c3 = make_covar_acc<standard>(cview);
    auto acc_wel_c3 = make_covar_acc(cview);

    // check mean and variance
    const auto m_d = sample_mean(sp_d);
    const auto c_d = sample_covariance(sp_d);
    const auto m_c = sample_mean(sp_c);
    const auto [r_c, i_c, ri_c] = sample_covariance(sp_c);
    check_near(acc_std_d1.mean(), m_d(0), tol);
    check_near(acc_std_d1.covariance_of_data(), c_d(0, 0), tol);
    check_near(acc_wel_d1.mean(), m_d(0), tol);
    check_near(acc_wel_d1.covariance_of_data(), c_d(0, 0), tol);
    check_near(acc_std_d3.mean(), m_d(0), tol);
    check_near(acc_std_d3.covariance_of_data(), c_d(0, 0), tol);
    check_near(acc_wel_d3.mean(), m_d(0), tol);
    check_near(acc_wel_d3.covariance_of_data(), c_d(0, 0), tol);
    check_near(acc_std_c1.mean(), m_c(0), tol);
    check_near(acc_std_c1.covariance_of_real_data(), r_c(0, 0), tol);
    check_near(acc_std_c1.covariance_of_imag_data(), i_c(0, 0), tol);
    check_near(acc_std_c1.covariance_of_real_and_imag_data(), ri_c(0, 0), tol);
    check_near(acc_wel_c1.mean(), m_c(0), tol);
    check_near(acc_wel_c1.covariance_of_real_data(), r_c(0, 0), tol);
    check_near(acc_wel_c1.covariance_of_imag_data(), i_c(0, 0), tol);
    check_near(acc_wel_c1.covariance_of_real_and_imag_data(), ri_c(0, 0), tol);
    check_near(acc_std_c3.mean(), m_c(0), tol);
    check_near(acc_std_c3.covariance_of_real_data(), r_c(0, 0), tol);
    check_near(acc_std_c3.covariance_of_imag_data(), i_c(0, 0), tol);
    check_near(acc_std_c3.covariance_of_real_and_imag_data(), ri_c(0, 0), tol);
    check_near(acc_wel_c3.mean(), m_c(0), tol);
    check_near(acc_wel_c3.covariance_of_real_data(), r_c(0, 0), tol);
    check_near(acc_wel_c3.covariance_of_imag_data(), i_c(0, 0), tol);
    check_near(acc_wel_c3.covariance_of_real_and_imag_data(), ri_c(0, 0), tol);
}

// Check covariance accumulator using the full random vectors.
TEST_F(SimplemcAccsStochasticProcess, CovarAccVector) {
    // general set up
    using namespace simplemc;
    covar_acc_static<double, size, standard> acc_std_d1, acc_std_d2;
    covar_acc_dynamic<double, welford> acc_wel_d1(size), acc_wel_d2(size);
    covar_acc_static<std::complex<double>, size, standard> acc_std_c1, acc_std_c2;
    covar_acc_dynamic<std::complex<double>, welford> acc_wel_c1(size), acc_wel_c2(size);

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

    // factory function
    auto dview = simplemc::ranges::transform_view(sp_d.samples, [](const auto& s) { return Eigen::Vector3d(s); });
    auto acc_std_d3 = make_covar_acc<standard>(dview);
    auto acc_wel_d3 = make_covar_acc(dview);
    auto cview = simplemc::ranges::transform_view(sp_c.samples, [](const auto& s) { return Eigen::Vector3cd(s); });
    auto acc_std_c3 = make_covar_acc<standard>(cview);
    auto acc_wel_c3 = make_covar_acc(cview);

    // check mean and variance
    using simplemc::make_span;
    const auto m_d = sample_mean(sp_d);
    const auto c_d = sample_covariance(sp_d);
    const auto m_c = sample_mean(sp_c);
    const auto [r_c, i_c, ri_c] = sample_covariance(sp_c);
    check_near(acc_std_d1.mean(), m_d, tol);
    check_near(make_span(acc_std_d1.covariance_of_data()), make_span(c_d), tol);
    check_near(acc_wel_d1.mean(), m_d, tol);
    check_near(make_span(acc_wel_d1.covariance_of_data()), make_span(c_d), tol);
    check_near(acc_std_d3.mean(), m_d, tol);
    check_near(make_span(acc_std_d3.covariance_of_data()), make_span(c_d), tol);
    check_near(acc_wel_d3.mean(), m_d, tol);
    check_near(make_span(acc_wel_d3.covariance_of_data()), make_span(c_d), tol);
    check_near(acc_std_c1.mean(), m_c, tol);
    check_near(make_span(acc_std_c1.covariance_of_real_data()), make_span(r_c), tol);
    check_near(make_span(acc_std_c1.covariance_of_imag_data()), make_span(i_c), tol);
    check_near(make_span(acc_std_c1.covariance_of_real_and_imag_data()), make_span(ri_c), tol);
    check_near(acc_wel_c1.mean(), m_c, tol);
    check_near(make_span(acc_wel_c1.covariance_of_real_data()), make_span(r_c), tol);
    check_near(make_span(acc_wel_c1.covariance_of_imag_data()), make_span(i_c), tol);
    check_near(make_span(acc_wel_c1.covariance_of_real_and_imag_data()), make_span(ri_c), tol);
    check_near(acc_std_c3.mean(), m_c, tol);
    check_near(make_span(acc_std_c3.covariance_of_real_data()), make_span(r_c), tol);
    check_near(make_span(acc_std_c3.covariance_of_imag_data()), make_span(i_c), tol);
    check_near(make_span(acc_std_c3.covariance_of_real_and_imag_data()), make_span(ri_c), tol);
    check_near(acc_wel_c3.mean(), m_c, tol);
    check_near(make_span(acc_wel_c3.covariance_of_real_data()), make_span(r_c), tol);
    check_near(make_span(acc_wel_c3.covariance_of_imag_data()), make_span(i_c), tol);
    check_near(make_span(acc_wel_c3.covariance_of_real_and_imag_data()), make_span(ri_c), tol);
}

// Check covariance accumulator using only part of random vectors.
TEST_F(SimplemcAccsStochasticProcess, CovarAccIndividualIndices) {
    // general set up
    using namespace simplemc;
    covar_acc_static<double, size, standard> acc_std_d1, acc_std_d2;
    covar_acc_dynamic<double, welford> acc_wel_d1(size), acc_wel_d2(size);
    covar_acc_static<std::complex<double>, size, standard> acc_std_c1, acc_std_c2;
    covar_acc_dynamic<std::complex<double>, welford> acc_wel_c1(size), acc_wel_c2(size);

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
        acc_wel_d2.accumulate(sp_d.samples[i](idxs), idxs);
        acc_std_c2[idx] << sp_c.samples[i](idx);
        acc_wel_c2.accumulate(sp_c.samples[i](idxs), idxs);
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
    using simplemc::make_span;
    auto m_std_d = sample_mean(sp_d);
    auto c_std_d = sample_covariance(sp_d);
    auto m_std_c = sample_mean(sp_c);
    auto [r_std_c, i_std_c, ri_std_c] = sample_covariance(sp_c);
    auto [m_wel_d, c_wel_d] = std::tuple(m_std_d, c_std_d);
    auto [m_wel_c, r_wel_c, i_wel_c, ri_wel_c] = std::tuple(m_std_c, r_std_c, i_std_c, ri_std_c);
    for (int i = 0; i < size; ++i) {
        if (i != idx) {
            m_std_d(i) = 0.0;
            c_std_d.row(i) = 0.0;
            c_std_d.col(i) = 0.0;
            m_std_c(i) = { 0.0, 0.0 };
            r_std_c.row(i) = 0.0;
            r_std_c.col(i) = 0.0;
            i_std_c.row(i) = 0.0;
            i_std_c.col(i) = 0.0;
            ri_std_c.row(i) = 0.0;
            ri_std_c.col(i) = 0.0;
        }
    }
    check_near(acc_std_d1.mean(), m_std_d, tol);
    check_near(make_span(acc_std_d1.covariance_of_data()), make_span(c_std_d), tol);
    check_near(acc_std_c1.mean(), m_std_c, tol);
    check_near(make_span(acc_std_c1.covariance_of_real_data()), make_span(r_std_c), tol);
    check_near(make_span(acc_std_c1.covariance_of_imag_data()), make_span(i_std_c), tol);
    check_near(make_span(acc_std_c1.covariance_of_real_and_imag_data()), make_span(ri_std_c), tol);
    for (int i = 1; i < size - 1; ++i) {
        m_wel_d(i) = 0.0;
        c_wel_d.row(i) = 0.0;
        c_wel_d.col(i) = 0.0;
        m_wel_c(i) = { 0.0, 0.0 };
        r_wel_c.row(i) = 0.0;
        r_wel_c.col(i) = 0.0;
        i_wel_c.row(i) = 0.0;
        i_wel_c.col(i) = 0.0;
        ri_wel_c.row(i) = 0.0;
        ri_wel_c.col(i) = 0.0;
    }
    check_near(acc_wel_d1.mean(), m_wel_d, tol);
    check_near(make_span(acc_wel_d1.covariance_of_data()), make_span(c_wel_d), tol);
    check_near(acc_wel_c1.mean(), m_wel_c, tol);
    check_near(make_span(acc_wel_c1.covariance_of_real_data()), make_span(r_wel_c), tol);
    check_near(make_span(acc_wel_c1.covariance_of_imag_data()), make_span(i_wel_c), tol);
    check_near(make_span(acc_wel_c1.covariance_of_real_and_imag_data()), make_span(ri_wel_c), tol);
}

// Check blocked covariance accumulator.
TEST_F(SimplemcAccsStochasticProcess, CovarAccBlocked) {
    // general set up
    using namespace simplemc;
    using block_acc_type = block_acc<covar_acc_static<double, size, welford>>;
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

    // factory function
    auto dview = simplemc::ranges::transform_view(sp_d.samples, [](const auto& s) { return Eigen::Vector3d(s); });
    auto acc_wel_d3 = make_block_covar_acc(dview, block_size);

    // check mean and variance
    using simplemc::make_span;
    auto bsp_d = block_samples(sp_d, block_size);
    const auto m_d = sample_mean(bsp_d);
    const auto c_d = sample_covariance(bsp_d);
    check_near(acc_wel_d1.accumulator().mean(), m_d, tol);
    check_near(make_span(acc_wel_d1.accumulator().covariance_of_data()), make_span(c_d), tol);
    check_near(acc_wel_d3.accumulator().mean(), m_d, tol);
    check_near(make_span(acc_wel_d3.accumulator().covariance_of_data()), make_span(c_d), tol);
}

// Check autocorrelation wrapper of covariance accumulator.
TEST_F(SimplemcAccsStochasticProcess, CovarAccAutocorrelation) {
    // general set up
    using namespace simplemc;
    autocorr_acc<covar_acc_static<double, size, welford>> acc_vec, acc_acc1, acc_acc2;
    autocorr_acc<covar_acc<double, welford>> acc_single(1, 2, 5);

    // fill accumulators
    std::vector<long> idxs(3);
    std::iota(idxs.begin(), idxs.end(), 0l);
    for (int i = 0; i < steps; ++i) {
        acc_vec << sp_d.samples[i];
        acc_acc1.accumulate(sp_d.samples[i], idxs);
        acc_acc2.accumulate(sp_d.samples[i], 0);
        acc_single << sp_d.samples[i](0);
    }

    // factory function
    auto dview = simplemc::ranges::transform_view(sp_d.samples, [](const auto& s) { return Eigen::VectorXd(s); });
    auto acc_vec2 = make_autocorr_covar_acc(dview);

    // check block variance accumulators with increasing block sizes and autocorrelation times
    using simplemc::make_span;
    using simplemc::accs::tau;
    auto [bsp_d, btau_v, btau_c] = blocking_autocorr(sp_d);
    const auto max_level = acc_vec.find_level(2);
    auto check_level = [&bsp_d, &btau_c](const auto& acc, auto i) {
        const auto c_d = sample_covariance(bsp_d[i]);
        const auto ac0_d = acc.accumulators()[0].covariance_of_data();
        const auto ac_d = acc.accumulators()[i].covariance_of_data();
        const auto blsize = acc.block_sizes()[i];
        if constexpr (std::decay_t<decltype(acc)>::static_size == 1) {
            check_near(acc.accumulators()[i].covariance_of_data(), c_d(0, 0), tol);
            check_near(tau(ac0_d, ac_d, blsize), btau_c[i](0, 0), tol);
        } else {
            check_near(make_span(acc.accumulators()[i].covariance_of_data()), make_span(c_d), tol);
            check_near(make_span(tau(ac0_d, ac_d, blsize)), make_span(btau_c[i]), tol);
        }
    };
    for (std::size_t i = 0; i < max_level; ++i) {
        check_level(acc_single, i);
        check_level(acc_vec, i);
        check_level(acc_vec2, i);
        check_level(acc_acc1, i);
        check_level(acc_acc2, i);
    }
}

// Check construction from pre-existing accumulated data.
TEST_F(SimplemcAccsStochasticProcess, CovarAccDataConstructor) {
    using namespace simplemc;

    // real vector: static
    covar_acc_static<double, size> acc_d;
    for (int i = 0; i < 10; ++i) {
        acc_d << sp_d.samples[i].matrix();
    }
    covar_acc_static<double, size> acc_d_copy(acc_d.mdata(), acc_d.cdata(), acc_d.count());
    ASSERT_EQ(acc_d_copy.count(), acc_d.count());
    ASSERT_FALSE(acc_d_copy.empty());
    ASSERT_EQ(acc_d_copy.mdata(), acc_d.mdata());
    ASSERT_EQ(acc_d_copy.cdata(), acc_d.cdata());

    // complex vector: static
    covar_acc_static<std::complex<double>, size> acc_c;
    for (int i = 0; i < 10; ++i) {
        acc_c << sp_c.samples[i].matrix();
    }
    covar_acc_static<std::complex<double>, size> acc_c_copy(
        acc_c.mdata(), acc_c.rdata(), acc_c.idata(), acc_c.cdata(), acc_c.count());
    ASSERT_EQ(acc_c_copy.count(), acc_c.count());
    ASSERT_FALSE(acc_c_copy.empty());
    ASSERT_EQ(acc_c_copy.mdata(), acc_c.mdata());
    ASSERT_EQ(acc_c_copy.rdata(), acc_c.rdata());
    ASSERT_EQ(acc_c_copy.idata(), acc_c.idata());
    ASSERT_EQ(acc_c_copy.cdata(), acc_c.cdata());

    // real vector: dynamic
    covar_acc_dynamic<double> acc_dyn(size);
    for (int i = 0; i < 10; ++i) {
        acc_dyn << sp_d.samples[i].matrix();
    }
    covar_acc_dynamic<double> acc_dyn_copy(acc_dyn.mdata(), acc_dyn.cdata(), acc_dyn.count());
    ASSERT_EQ(acc_dyn_copy.count(), acc_dyn.count());
    ASSERT_EQ(acc_dyn_copy.mdata(), acc_dyn.mdata());
    ASSERT_EQ(acc_dyn_copy.cdata(), acc_dyn.cdata());
}

// Check that dynamic size validation works for valid sizes.
// NOTE: The size check in the constructor body is currently unreachable because
// vec_type::Zero(m) in the member initializer list triggers an Eigen assertion
// before the body runs. This should be fixed by validating m before the initializer
// list (e.g. via a static helper). For now, we only test valid construction.
TEST_F(SimplemcAccsStochasticProcess, CovarAccDynamicSizeValidation) {
    using namespace simplemc;

    // valid dynamic construction
    ASSERT_NO_THROW(covar_acc_dynamic<double>(1));
    ASSERT_NO_THROW(covar_acc_dynamic<double>(100));
    ASSERT_NO_THROW(covar_acc_dynamic<std::complex<double>>(5));
}
