#include "./accs_fixture.hpp"

#include <simplemc/accs/mean_acc.hpp>

#include <complex>
#include <numeric>
#include <vector>

// anonymous namespace with parameters
namespace {

using namespace simplemc;
constexpr auto standard = accs::varalg::standard;
constexpr auto welford = accs::varalg::welford;
constexpr double tol = 1e-10;

} // namespace

// Check empty accumulators.
TEST_F(SimplemcAccs, MeanAccEmpty) {
    using namespace simplemc;

    mean_acc_single<double> acc_sd;
    ASSERT_EQ(acc_sd.size(), 1);
    check_empty(acc_sd);
    static_assert(!acc_sd.is_dynamic);
    static_assert(acc_sd.static_size == 1);

    mean_acc_single<std::complex<double>, standard> acc_sc;
    ASSERT_EQ(acc_sc.size(), 1);
    check_empty(acc_sc);
    static_assert(!acc_sc.is_dynamic);
    static_assert(acc_sc.static_size == 1);

    mean_acc_static<double, 5> acc_st_d;
    ASSERT_EQ(acc_st_d.size(), 5);
    check_empty(acc_st_d);
    static_assert(!acc_st_d.is_dynamic);
    static_assert(acc_st_d.static_size == 5);

    mean_acc_static<std::complex<double>, 5, standard> acc_st_c;
    ASSERT_EQ(acc_st_c.size(), 5);
    check_empty(acc_st_c);
    static_assert(!acc_st_c.is_dynamic);
    static_assert(acc_st_c.static_size == 5);

    mean_acc_dynamic<double> acc_dyn_d(5);
    ASSERT_EQ(acc_dyn_d.size(), 5);
    check_empty(acc_dyn_d);
    static_assert(acc_dyn_d.is_dynamic);
    static_assert(acc_dyn_d.static_size == Eigen::Dynamic);

    mean_acc_dynamic<std::complex<double>, standard> acc_dyn_c(5);
    ASSERT_EQ(acc_dyn_c.size(), 5);
    check_empty(acc_dyn_c);
    static_assert(acc_dyn_c.is_dynamic);
    static_assert(acc_dyn_c.static_size == Eigen::Dynamic);
}

// Check mean accumulator of size 1.
TEST_F(SimplemcAccs, MeanAccSingle) {
    // general set up
    using namespace simplemc;
    using dbl_acc_type = simplemc::mean_acc_single<double, standard>;
    using cplx_acc_type = simplemc::mean_acc_single<std::complex<double>, welford>;
    dbl_acc_type acc_d1, acc_d2;
    cplx_acc_type acc_c1, acc_c2;

    // fill accumulators
    const auto merge_size = steps / 2;
    for (int i = 0; i < merge_size; ++i) {
        acc_d1 << sp_d.samples[i](0);
        acc_c1 << sp_c.samples[i](0);
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_d2[0] << sp_d.samples[i](0);
        acc_c2[0] << sp_c.samples[i](0);
    }

    // check count
    ASSERT_EQ(acc_d1.count(), merge_size);
    ASSERT_EQ(acc_c1.count(), merge_size);
    ASSERT_EQ(acc_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_c2.count(), steps - merge_size);

    // merge accumulators
    acc_d1 << acc_d2;
    acc_c1 << acc_c2;

    // check mean
    const auto m_d = sample_mean(sp_d);
    const auto m_c = sample_mean(sp_c);
    check_near(acc_d1.mean(), m_d[0], tol);
    check_near(acc_c1.mean(), m_c[0], tol);
}

// Check mean accumulator using the full random vectors.
TEST_F(SimplemcAccs, MeanAccVector) {
    // general set up
    using namespace simplemc;
    using dbl_acc_type = simplemc::mean_acc_static<double, size, standard>;
    using cplx_acc_type = simplemc::mean_acc_dynamic<std::complex<double>, welford>;
    dbl_acc_type acc_d1, acc_d2;
    cplx_acc_type acc_c1(size), acc_c2(size);

    // fill accumulators
    const auto merge_size = steps / 2;
    std::vector<long> idxs(3);
    std::iota(idxs.begin(), idxs.end(), 0l);
    for (int i = 0; i < merge_size; ++i) {
        acc_d1 << sp_d.samples[i].matrix();
        acc_c1.accumulate(sp_c.samples[i], 0);
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_d2.accumulate(sp_d.samples[i], idxs);
        acc_c2 << sp_c.samples[i].matrix();
    }

    // check count
    ASSERT_EQ(acc_d1.count(), merge_size);
    ASSERT_EQ(acc_c1.count(), merge_size);
    ASSERT_EQ(acc_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_c2.count(), steps - merge_size);

    // merge accumulators
    acc_d1 << acc_d2;
    acc_c1 << acc_c2;

    // check mean
    const auto m_d = sample_mean(sp_d);
    const auto m_c = sample_mean(sp_c);
    check_range_near(acc_d1.mean(), m_d, tol);
    check_range_near(acc_c1.mean(), m_c, tol);
}

// Check mean accumulator using the full random vectors.
TEST_F(SimplemcAccs, MeanAccIndividualIndices) {
    // general set up
    using namespace simplemc;
    using dbl_acc_type = simplemc::mean_acc_static<double, size, standard>;
    using cplx_acc_type = simplemc::mean_acc_dynamic<std::complex<double>, welford>;
    dbl_acc_type acc_d1, acc_d2;
    cplx_acc_type acc_c1(size), acc_c2(size);

    // fill accumulators
    const auto merge_size = steps / 2;
    std::vector<long> idxs { 0, size - 1 };
    for (int i = 0; i < merge_size; ++i) {
        acc_d1[1] << sp_d.samples[i](1);
        acc_c1.accumulate(sp_c.samples[i](idxs), idxs);
    }
    for (int i = merge_size; i < steps; ++i) {
        acc_d2[1] << sp_d.samples[i](1);
        auto mva_c = acc_c2.make_mva();
        mva_c[idxs[0]] << sp_c.samples[i](idxs[0]);
        mva_c[idxs[1]] << sp_c.samples[i](idxs[1]);
        mva_c.increment_count();
    }

    // check count
    ASSERT_EQ(acc_d1.count(), merge_size);
    ASSERT_EQ(acc_c1.count(), merge_size);
    ASSERT_EQ(acc_d2.count(), steps - merge_size);
    ASSERT_EQ(acc_c2.count(), steps - merge_size);

    // merge accumulators
    acc_d1 << acc_d2;
    acc_c1 << acc_c2;

    // check mean
    const auto m_d = sample_mean(sp_d);
    auto m_c = sample_mean(sp_c);
    for (int i = 1; i < size - 1; ++i) m_c(i) = { 0.0, 0.0 };
    check_near(acc_d1.mean()(1), m_d(1), tol);
    check_range_near(acc_c1.mean(), m_c, tol);
}
