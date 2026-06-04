#ifndef SIMPLEMC_TESTS_SERIALIZE_SERIALIZE_TEST_UTILS_HPP
#define SIMPLEMC_TESTS_SERIALIZE_SERIALIZE_TEST_UTILS_HPP

#include "../test_utils.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/grids/custom_grid.hpp>
#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/nd_grid.hpp>
#include <simplemc/grids/power_grid.hpp>
#include <simplemc/grids/symmetric_power_grid.hpp>
#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/numeric/eigen.hpp>

#include <complex>
#include <concepts>
#include <cstddef>
#include <utility>

// Check that two mean_acc instances have equal state.
template <class T, simplemc::varalg A>
void check_state_equal(const simplemc::mean_acc<T, A>& a, const simplemc::mean_acc<T, A>& b) {
    EXPECT_EQ(a.count(), b.count());
    check_near(a.mdata(), b.mdata());
}

// Check that two var_acc instances have equal state.
template <class T, simplemc::varalg A>
void check_state_equal(const simplemc::var_acc<T, A>& a, const simplemc::var_acc<T, A>& b) {
    EXPECT_EQ(a.count(), b.count());
    check_near(a.mdata(), b.mdata());
    check_near(a.cdata(), b.cdata());
    if constexpr (std::same_as<T, std::complex<double>> || simplemc::eigen_vector_cplx<T>) {
        check_near(a.rdata(), b.rdata());
        check_near(a.idata(), b.idata());
    }
}

// Check that two covar_acc instances have equal state.
template <class T, simplemc::varalg A>
void check_state_equal(const simplemc::covar_acc<T, A>& a, const simplemc::covar_acc<T, A>& b) {
    EXPECT_EQ(a.count(), b.count());
    check_near(a.mdata(), b.mdata());
    check_near(a.cdata(), b.cdata());
    if constexpr (std::same_as<T, std::complex<double>> || simplemc::eigen_vector_cplx<T>) {
        check_near(a.rdata(), b.rdata());
        check_near(a.idata(), b.idata());
    }
}

// Check that two batch_acc instances have equal state.
template <class T, simplemc::varalg A>
void check_state_equal(const simplemc::batch_acc<T, A>& a, const simplemc::batch_acc<T, A>& b) {
    const auto& a_full = a.batch_vector_full();
    const auto& b_full = b.batch_vector_full();
    const auto& a_acc = a.batch_vector_accumulating();
    const auto& b_acc = b.batch_vector_accumulating();
    ASSERT_EQ(a_full.size(), b_full.size());
    ASSERT_EQ(a_acc.size(), b_acc.size());
    for (std::size_t i = 0; i < a_full.size(); ++i) {
        check_state_equal(a_full[i], b_full[i]);
        check_state_equal(a_acc[i], b_acc[i]);
    }
}

// Check that two block_acc instances have equal state.
template <class Acc>
void check_state_equal(const simplemc::block_acc<Acc>& a, const simplemc::block_acc<Acc>& b) {
    EXPECT_EQ(a.block_size(), b.block_size());
    check_state_equal(a.accumulator(), b.accumulator());
    check_state_equal(a.block(), b.block());
}

// Check that two autocorr_acc instances have equal state.
template <class Acc>
void check_state_equal(const simplemc::autocorr_acc<Acc>& a, const simplemc::autocorr_acc<Acc>& b) {
    EXPECT_EQ(a.factor(), b.factor());
    EXPECT_EQ(a.min_levels(), b.min_levels());
    ASSERT_EQ(a.num_levels(), b.num_levels());
    for (std::size_t l = 0; l < a.num_levels(); ++l) {
        check_state_equal(a.accumulators()[l], b.accumulators()[l]);
        check_state_equal(a.blocks()[l], b.blocks()[l]);
    }
}

// Check that two linear_grid instances have equal state.
inline void check_state_equal(const simplemc::linear_grid& a, const simplemc::linear_grid& b) {
    EXPECT_DOUBLE_EQ(a.first(), b.first());
    EXPECT_DOUBLE_EQ(a.last(), b.last());
    EXPECT_EQ(a.size(), b.size());
}

// Check that two power_grid instances have equal state.
inline void check_state_equal(const simplemc::power_grid& a, const simplemc::power_grid& b) {
    EXPECT_DOUBLE_EQ(a.first(), b.first());
    EXPECT_DOUBLE_EQ(a.last(), b.last());
    EXPECT_EQ(a.size(), b.size());
    EXPECT_DOUBLE_EQ(a.power(), b.power());
}

// Check that two symmetric_power_grid instances have equal state.
inline void check_state_equal(const simplemc::symmetric_power_grid& a, const simplemc::symmetric_power_grid& b) {
    EXPECT_DOUBLE_EQ(a.first(), b.first());
    EXPECT_DOUBLE_EQ(a.last(), b.last());
    EXPECT_EQ(a.size(), b.size());
    EXPECT_DOUBLE_EQ(a.grid1().power(), b.grid1().power());
}

// Check that two custom_grid instances have equal state.
inline void check_state_equal(const simplemc::custom_grid& a, const simplemc::custom_grid& b) {
    check_near(a.grid_points(), b.grid_points());
}

// Check that two nd_grid instances have equal state.
template <class... Grids>
void check_state_equal(const simplemc::nd_grid<Grids...>& a, const simplemc::nd_grid<Grids...>& b) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (check_state_equal(std::get<I>(a.grids()), std::get<I>(b.grids())), ...);
    }(std::index_sequence_for<Grids...> {});
}

// Check that two bravais_lattice instances have equal state.
template <int N>
void check_state_equal(const simplemc::bravais_lattice<N>& a, const simplemc::bravais_lattice<N>& b) {
    check_near(a.real_lattice_vectors(), b.real_lattice_vectors());
}

#endif // SIMPLEMC_TESTS_SERIALIZE_SERIALIZE_TEST_UTILS_HPP
