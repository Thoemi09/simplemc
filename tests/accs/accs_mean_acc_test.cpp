#include "./accs_fixture.hpp"

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/utils/ranges.hpp>

#include <complex>
#include <numeric>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = varalg::standard;
constexpr auto welford = varalg::welford;
constexpr double tol = 1e-10;

} // namespace

// Test accumulator concepts.
TEST_F(SimplemcAccs, MeanAccConcepts) {
    // basic_accumulator concept for mean_acc
    static_assert(basic_accumulator<mean_acc<double>>);
    static_assert(basic_accumulator<mean_acc<std::complex<double>>>);
    static_assert(basic_accumulator<mean_acc<Eigen::VectorXd>>);

    // basic_accumulator concept for multivalue_acc<mean_acc>
    static_assert(basic_accumulator<multivalue_acc<mean_acc<double>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc<std::complex<double>>>>);
    static_assert(basic_accumulator<multivalue_acc<mean_acc<Eigen::Vector3d>>>);

    // mean_accumulator concept for mean_acc
    static_assert(mean_accumulator<mean_acc<double>>);
    static_assert(mean_accumulator<mean_acc<std::complex<double>>>);
    static_assert(mean_accumulator<mean_acc<Eigen::VectorXd>>);
}

// Check empty accumulators.
TEST_F(SimplemcAccs, MeanAccEmpty) {
    using namespace simplemc;

    mean_acc<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_empty(acc_sd);
    acc_sd << acc_sd;
    check_empty(acc_sd);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    mean_acc<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_empty(acc_sc);
    acc_sc << acc_sc;
    check_empty(acc_sc);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    mean_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_empty(acc_st_d);
    acc_st_d << acc_st_d;
    check_empty(acc_st_d);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    mean_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_empty(acc_st_c);
    acc_st_c << acc_st_c;
    check_empty(acc_st_c);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    mean_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_empty(acc_dyn_d);
    acc_dyn_d << acc_dyn_d;
    check_empty(acc_dyn_d);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    mean_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_empty(acc_dyn_c);
    acc_dyn_c << acc_dyn_c;
    check_empty(acc_dyn_c);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Check mean accumulator of size 1.
TEST_F(SimplemcAccs, MeanAccSingle) {
    // general set up
    using namespace simplemc;
    mean_acc<double, standard> acc_std_d1, acc_std_d2;
    mean_acc<double, welford> acc_wel_d1, acc_wel_d2;
    mean_acc<std::complex<double>, standard> acc_std_c1, acc_std_c2;
    mean_acc<std::complex<double>, welford> acc_wel_c1, acc_wel_c2;

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
    auto acc_std_d3 = make_mean_acc<standard>(dview);
    auto acc_wel_d3 = make_mean_acc(dview);
    auto cview = simplemc::ranges::transform_view(sp_c.samples, [](const auto& s) { return s(0); });
    auto acc_std_c3 = make_mean_acc<standard>(cview);
    auto acc_wel_c3 = make_mean_acc(cview);

    // check mean
    const auto m_d = sample_mean(sp_d);
    const auto m_c = sample_mean(sp_c);
    check_near(acc_std_d1.mean(), m_d[0], tol);
    check_near(acc_wel_d1.mean(), m_d[0], tol);
    check_near(acc_std_d3.mean(), m_d[0], tol);
    check_near(acc_wel_d3.mean(), m_d[0], tol);
    check_near(acc_std_c1.mean(), m_c[0], tol);
    check_near(acc_wel_c1.mean(), m_c[0], tol);
    check_near(acc_std_c3.mean(), m_c[0], tol);
    check_near(acc_wel_c3.mean(), m_c[0], tol);

    // reset and check empty again
    acc_std_d1.reset();
    acc_wel_d1.reset();
    acc_std_c1.reset();
    acc_wel_c1.reset();
    check_empty(acc_std_d1);
    check_empty(acc_wel_d1);
    check_empty(acc_std_c1);
    check_empty(acc_wel_c1);
}

// Check mean accumulator using the full random vectors.
TEST_F(SimplemcAccs, MeanAccVector) {
    // general set up
    using namespace simplemc;
    mean_acc_static<double, size, standard> acc_std_d1, acc_std_d2;
    mean_acc_dynamic<double, welford> acc_wel_d1(size), acc_wel_d2(size);
    mean_acc_static<std::complex<double>, size, standard> acc_std_c1, acc_std_c2;
    mean_acc_dynamic<std::complex<double>, welford> acc_wel_c1(size), acc_wel_c2(size);

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
    auto acc_std_d3 = make_mean_acc<standard>(dview);
    auto acc_wel_d3 = make_mean_acc(dview);
    auto cview = simplemc::ranges::transform_view(sp_c.samples, [](const auto& s) { return Eigen::Vector3cd(s); });
    auto acc_std_c3 = make_mean_acc<standard>(cview);
    auto acc_wel_c3 = make_mean_acc(cview);

    // check mean
    const auto m_d = sample_mean(sp_d);
    const auto m_c = sample_mean(sp_c);
    check_range_near(acc_std_d1.mean(), m_d, tol);
    check_range_near(acc_wel_d1.mean(), m_d, tol);
    check_range_near(acc_std_d3.mean(), m_d, tol);
    check_range_near(acc_wel_d3.mean(), m_d, tol);
    check_range_near(acc_std_c1.mean(), m_c, tol);
    check_range_near(acc_wel_c1.mean(), m_c, tol);
    check_range_near(acc_std_c3.mean(), m_c, tol);
    check_range_near(acc_wel_c3.mean(), m_c, tol);

    // reset and check empty again
    acc_std_d1.reset();
    acc_wel_d1.reset();
    acc_std_c1.reset();
    acc_wel_c1.reset();
    check_empty(acc_std_d1);
    check_empty(acc_wel_d1);
    check_empty(acc_std_c1);
    check_empty(acc_wel_c1);
}

// Check mean accumulator using only part of random vectors.
TEST_F(SimplemcAccs, MeanAccIndividualIndices) {
    // general set up
    using namespace simplemc;
    mean_acc_static<double, size, standard> acc_std_d1, acc_std_d2;
    mean_acc_dynamic<double, welford> acc_wel_d1(size), acc_wel_d2(size);
    mean_acc_static<std::complex<double>, size, standard> acc_std_c1, acc_std_c2;
    mean_acc_dynamic<std::complex<double>, welford> acc_wel_c1(size), acc_wel_c2(size);

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
        for (auto idx : idxs) {
            mva_d[idx] << sp_d.samples[i](idx);
            mva_c[idx] << sp_c.samples[i](idx);
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

    // check mean
    const auto m_d = sample_mean(sp_d);
    const auto m_c = sample_mean(sp_c);
    auto m_wel_d = m_d;
    auto m_wel_c = m_c;
    for (int i = 1; i < size - 1; ++i) {
        m_wel_d(i) = 0.0;
        m_wel_c(i) = { 0.0, 0.0 };
    }
    check_near(acc_std_d1.mean()(idx), m_d(idx), tol);
    check_range_near(acc_wel_d1.mean(), m_wel_d, tol);
    check_near(acc_std_c1.mean()(idx), m_c(idx), tol);
    check_range_near(acc_wel_c1.mean(), m_wel_c, tol);
}
