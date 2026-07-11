// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#ifndef SIMPLEMC_TESTS_ACCS_TEST_TRAITS_HPP
#define SIMPLEMC_TESTS_ACCS_TEST_TRAITS_HPP

#include "../test_utils.hpp"

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/batch_acc.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <type_traits>

namespace test_detail {

using namespace simplemc;

// Tag type carrying sample type T and varalg A.
template <typename T, varalg A>
struct AccTag {
    using sample_t = T;
    static constexpr varalg alg = A;
};

// Type list: 12 combinations (6 sample types x 2 varalg).
using AllAccTypes = ::testing::Types<AccTag<double, varalg::standard>, AccTag<double, varalg::welford>,
    AccTag<std::complex<double>, varalg::standard>, AccTag<std::complex<double>, varalg::welford>,
    AccTag<Eigen::Vector3d, varalg::standard>, AccTag<Eigen::Vector3d, varalg::welford>,
    AccTag<Eigen::Vector3cd, varalg::standard>, AccTag<Eigen::Vector3cd, varalg::welford>,
    AccTag<Eigen::VectorXd, varalg::standard>, AccTag<Eigen::VectorXd, varalg::welford>,
    AccTag<Eigen::VectorXcd, varalg::standard>, AccTag<Eigen::VectorXcd, varalg::welford>>;

// Sample type classification.
template <typename T>
constexpr bool is_complex_sample_v = std::is_same_v<T, std::complex<double>> || eigen_vector_cplx<T>;

template <typename T>
constexpr bool is_scalar_sample_v = sample_scalar<T>;

template <typename T>
constexpr bool is_dynamic_sample_v = std::is_same_v<T, Eigen::VectorXd> || std::is_same_v<T, Eigen::VectorXcd>;

// Vector size: 1 for scalar, 3 for all vector types.
template <typename T>
constexpr long vec_size_v = is_scalar_sample_v<T> ? 1L : 3L;

// Real vector/scalar type.
template <typename T>
using real_vec_t = std::conditional_t<is_scalar_sample_v<T>, double,
    Eigen::Matrix<double, (is_dynamic_sample_v<T> ? Eigen::Dynamic : vec_size_v<T>), 1>>;

// Real matrix/scalar type for covariance.
template <typename T>
using real_mat_t = std::conditional_t<is_scalar_sample_v<T>, double,
    Eigen::Matrix<double, (is_dynamic_sample_v<T> ? Eigen::Dynamic : vec_size_v<T>),
        (is_dynamic_sample_v<T> ? Eigen::Dynamic : vec_size_v<T>)>>;

// Construct a zero-initialized value of type T with the right size.
template <typename T>
auto make_zero_sample() {
    if constexpr (is_scalar_sample_v<T>) {
        return T { 0 };
    } else if constexpr (is_dynamic_sample_v<T>) {
        return T::Zero(vec_size_v<T>);
    } else {
        return T::Zero();
    }
}

// Build a real_vec_t<T> from a per-component function f(j).
template <typename T, typename F>
auto make_real_vec(F f) {
    constexpr long M = vec_size_v<T>;
    if constexpr (is_scalar_sample_v<T>) {
        return f(0L);
    } else {
        real_vec_t<T> v = real_vec_t<T>::Zero(M);
        for (long j = 0; j < M; ++j) {
            v(j) = f(j);
        }
        return v;
    }
}

// Build a real_mat_t<T> from a per-element function f(j, k).
template <typename T, typename F>
auto make_real_mat(F f) {
    constexpr long M = vec_size_v<T>;
    if constexpr (is_scalar_sample_v<T>) {
        return f(0L, 0L);
    } else {
        real_mat_t<T> m(M, M);
        for (long j = 0; j < M; ++j) {
            for (long k = 0; k < M; ++k) {
                m(j, k) = f(j, k);
            }
        }
        return m;
    }
}

// ---------------------------------------------------------------------------
// Deterministic dataset and expected values.
//
// N = 100 samples indexed by i = 0..99, with M-dimensional components indexed by j = 0..M-1.
// The general sample formula is:
//
//   re(i, j) = (j + 1) * (i + 1)         -- real part of component j of sample i
//   im(i, j) = (j + 1) * (100 - i)       -- imaginary part (complex types only)
//
// Concrete sample types:
//
//   Scalar double  (M=1):  x[i] = i + 1                                 (values 1..100)
//   Scalar complex (M=1):  z[i] = (i+1) + (100-i)*1i
//   Vector double  (M=3):  v[i](j) = (j+1)*(i+1)
//   Vector complex (M=2):  v[i](j) = (j+1)*(i+1) + (j+1)*(100-i)*1i
//
// Expected values derivation:
//
// All components are arithmetic sequences in i, so closed-form results apply.
//
// Mean:
//   E[re(j)] = (j+1) * E[i+1] = (j+1) * (N+1)/2 = (j+1) * 50.5
//   E[im(j)] = (j+1) * E[100-i] = (j+1) * (N+1)/2 = (j+1) * 50.5
//
// Variance (Bessel-corrected, i.e. dividing by N-1):
//   For the arithmetic sequence a_i = a + d*i with N terms:
//     Var = d^2 * N * (N-1) / 12 * 1/(N-1) = d^2 * N / 12  ... (wrong)
//   More carefully: sum_{i=0}^{N-1} (a_i - mean)^2 = d^2 * N*(N^2-1)/12.
//   Bessel-corrected: V_base = d^2 * N*(N^2-1) / (12*(N-1)) = d^2 * N*(N+1)/12.
//   For d=1 (scalar): V_base = N*(N+1)/12 = 100*101/12 = 2525/3.
//   For component j with d=(j+1): Var(j) = (j+1)^2 * V_base.
//
// Covariance:
//   Components j and k are perfectly correlated (both linear in i):
//     Cov(re_j, re_k) = (j+1)*(k+1) * V_base.
//   The imaginary part im(j) = (j+1)*(100-i) is a reversed arithmetic sequence with the same
//   spread, so Var(im_j) = (j+1)^2 * V_base = Var(re_j).
//
// Cross-covariance (real-imag):
//   Let u = i+1 - mean = i+1 - 50.5 and note that (100-i) - 50.5 = -u.
//   So Cov(re_j, im_k) = (j+1)*(k+1) * E[u * (-u)] = -(j+1)*(k+1) * V_base.
// ---------------------------------------------------------------------------

constexpr int N = 100;
constexpr double base_var = 2525.0 / 3.0; // N*(N+1)/12 = 100*101/12

// Build a sample of type T from per-component real/imag functions.
// For real types, im is unused.
template <typename T, typename Re, typename Im>
auto build_sample(Re re, Im im) {
    if constexpr (std::is_same_v<T, double>) {
        return re(0L);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return T { re(0L), im(0L) };
    } else {
        T v = make_zero_sample<T>();
        for (long j = 0; j < vec_size_v<T>; ++j) {
            if constexpr (is_complex_sample_v<T>) {
                v(j) = typename T::Scalar(re(j), im(j));
            } else {
                v(j) = re(j);
            }
        }
        return v;
    }
}

// Generate one deterministic sample of type T for index i.
template <typename T>
auto make_sample(int i) {
    return build_sample<T>([&](long j) { return static_cast<double>((j + 1) * (i + 1)); },
        [&](long j) { return static_cast<double>((j + 1) * (100 - i)); });
}

// Generate N deterministic samples for type T.
template <typename T>
auto make_data() {
    std::vector<T> data;
    data.reserve(N);
    for (int i = 0; i < N; ++i) {
        data.push_back(make_sample<T>(i));
    }
    return data;
}

// E[re(j)] = E[im(j)] = (j+1) * 50.5.
template <typename T>
auto expected_mean() {
    return build_sample<T>([](long j) { return static_cast<double>(j + 1) * 50.5; },
        [](long j) { return static_cast<double>(j + 1) * 50.5; });
}

// Var(j) = (j+1)^2 * V_base  (Bessel-corrected).
template <typename T>
auto expected_var_data() {
    return make_real_vec<T>([](long j) { return static_cast<double>((j + 1) * (j + 1)) * base_var; });
}

// Var(re_j) = Var(im_j) = (j+1)^2 * V_base  (reversed sequence has same variance).
template <typename T>
auto expected_var_real_data() {
    return expected_var_data<T>();
}

template <typename T>
auto expected_var_imag_data() {
    return expected_var_data<T>();
}

// Cov(re_j, im_j) = -(j+1)^2 * V_base  (see cross-covariance derivation above).
template <typename T>
auto expected_cov_re_im_data() {
    return make_real_vec<T>([](long j) { return -static_cast<double>((j + 1) * (j + 1)) * base_var; });
}

// Cov(re_j, re_k) = (j+1)*(k+1) * V_base.
template <typename T>
auto expected_covar_data() {
    return make_real_mat<T>([](long j, long k) { return static_cast<double>((j + 1) * (k + 1)) * base_var; });
}

// Cov(rr) = Cov(ii) = (j+1)*(k+1) * V_base.
template <typename T>
auto expected_cov_rr() {
    return expected_covar_data<T>();
}

template <typename T>
auto expected_cov_ii() {
    return expected_covar_data<T>();
}

// Cov(re_j, im_k) = -(j+1)*(k+1) * V_base.
template <typename T>
auto expected_cov_ri() {
    return make_real_mat<T>([](long j, long k) { return -static_cast<double>((j + 1) * (k + 1)) * base_var; });
}

// Construct an empty accumulator of the given type, handling dynamic size.
template <typename A, typename T>
auto make_empty() {
    if constexpr (is_dynamic_sample_v<T>) {
        return A { vec_size_v<T> };
    } else {
        return A {};
    }
}

// Construct an empty batch accumulator (needs size and m_b).
template <typename A, typename T>
auto make_empty_batch(std::size_t m_b = 256) {
    return A { vec_size_v<T>, m_b };
}

// Construct an empty block accumulator (needs block_size and size).
template <typename A, typename T>
auto make_empty_block(std::uint64_t b) {
    return A { b, vec_size_v<T> };
}

// Construct an empty autocorr accumulator (needs size, factor, min_levels).
template <typename A, typename T>
auto make_empty_autocorr(std::size_t factor = 2, std::size_t min_levels = 2) {
    return A { vec_size_v<T>, factor, min_levels };
}

// Extract component from sample (works for both scalar and vector).
template <typename T>
auto component(const T& sample, long idx) {
    if constexpr (is_scalar_sample_v<T>) {
        (void)idx;
        return sample;
    } else {
        return sample(idx);
    }
}

// Create a shift value for factory tests.
template <typename T>
auto make_shift() {
    return build_sample<T>([](long) { return 1.0; }, [](long) { return 1.0; });
}

// Check an empty accumulator (shared with basic and advanced accumulator tests).
template <typename A>
void check_acc_empty(const A& acc) {
    ASSERT_EQ(acc.count(), 0);
    ASSERT_TRUE(acc.empty());
    check_isnan(acc.mean());
}

// Check all available statistics of an accumulator against expected values.
// Uses requires-expressions to detect the accumulator kind:
//   - mean_acc:  checks mean only
//   - var_acc:   checks mean + variance (+ real/imag components for complex)
//   - covar_acc: checks mean + covariance matrix (+ component matrices for complex)
template <typename T, typename Acc>
void check_acc_results(const Acc& acc, double eps_mean = 1e-10, double eps_stat = 1e-8) {
    // Mean (all accumulator types).
    ASSERT_EQ(acc.count(), N);
    ASSERT_FALSE(acc.empty());
    check_near(acc.mean(), expected_mean<T>(), eps_mean);

    // Variance details (var_acc only — detected via variance_of_data).
    if constexpr (requires { acc.variance_of_data(); }) {
        if constexpr (is_complex_sample_v<T>) {
            check_near(acc.variance_of_real_data(), expected_var_real_data<T>(), eps_stat);
            check_near(acc.variance_of_imag_data(), expected_var_imag_data<T>(), eps_stat);
            check_near(acc.covariance_of_real_and_imag_data(), expected_cov_re_im_data<T>(), eps_stat);
            real_vec_t<T> total_var = expected_var_real_data<T>() + expected_var_imag_data<T>();
            check_near(acc.variance_of_data(), total_var, eps_stat);
            real_vec_t<T> var_mean = total_var / static_cast<double>(N);
            check_near(acc.variance(), var_mean, eps_stat);
        } else {
            check_near(acc.variance_of_data(), expected_var_data<T>(), eps_stat);
            real_vec_t<T> var_mean = expected_var_data<T>() / static_cast<double>(N);
            check_near(acc.variance(), var_mean, eps_stat);
        }
    }

    // Covariance details (covar_acc only — detected via covariance_of_data).
    if constexpr (requires { acc.covariance_of_data(); }) {
        if constexpr (is_complex_sample_v<T>) {
            check_near(acc.covariance_of_real_data(), expected_cov_rr<T>(), eps_stat);
            check_near(acc.covariance_of_imag_data(), expected_cov_ii<T>(), eps_stat);
            check_near(acc.covariance_of_real_and_imag_data(), expected_cov_ri<T>(), eps_stat);
            if constexpr (is_scalar_sample_v<T>) {
                double total_cov = expected_cov_rr<T>() + expected_cov_ii<T>();
                check_near(acc.covariance_of_data(), total_cov, eps_stat);
            }
        } else {
            check_near(acc.covariance_of_data(), expected_covar_data<T>(), eps_stat);
            if constexpr (is_scalar_sample_v<T>) {
                double cov_mean = expected_covar_data<T>() / static_cast<double>(N);
                check_near(acc.covariance(), cov_mean, eps_stat);
                check_near(acc.variance(), cov_mean, eps_stat);
            } else {
                real_mat_t<T> cov_mean = expected_covar_data<T>() / static_cast<double>(N);
                check_near(acc.covariance(), cov_mean, eps_stat);
                real_vec_t<T> var_mean = expected_var_data<T>() / static_cast<double>(N);
                check_near(acc.variance(), var_mean, eps_stat);
            }
        }
    }
}

// Compare two accumulators' internal data.
// Uses requires-expressions to detect available accessors (mdata, cdata, rdata, idata,
// batch_vector_full/accumulating).
template <typename Acc>
void check_acc_equal(const Acc& acc, const Acc& expected, double eps = 1e-14) {
    ASSERT_EQ(acc.count(), expected.count());

    if constexpr (requires { acc.mdata(); }) {
        check_near(acc.mdata(), expected.mdata(), eps);
    }
    if constexpr (requires { acc.cdata(); }) {
        check_near(acc.cdata(), expected.cdata(), eps);
    }
    if constexpr (requires { acc.rdata(); }) {
        check_near(acc.rdata(), expected.rdata(), eps);
        check_near(acc.idata(), expected.idata(), eps);
    }
    if constexpr (requires { acc.batch_vector_full(); }) {
        const auto& full = acc.batch_vector_full();
        const auto& full_exp = expected.batch_vector_full();
        const auto& accum = acc.batch_vector_accumulating();
        const auto& accum_exp = expected.batch_vector_accumulating();
        ASSERT_EQ(full.size(), full_exp.size());
        ASSERT_EQ(accum.size(), accum_exp.size());
        ASSERT_EQ(accum.size(), accum_exp.size());
        for (std::size_t i = 0; i < full.size(); ++i) {
            check_acc_equal(full[i], full_exp[i], eps);
            check_acc_equal(accum[i], accum_exp[i], eps);
        }
    }
}

// Check that a block_acc has the expected structure and data.
template <typename A>
void check_block_acc(const A& acc, std::uint64_t bl_size, double tol = 1e-10) {
    using acc_t = typename A::acc_type;
    using sample_t = typename acc_t::sample_type;

    // structural checks
    ASSERT_EQ(acc.block_size(), bl_size);
    ASSERT_EQ(acc.count(), N / bl_size);
    ASSERT_EQ(acc.total_count(), N);
    ASSERT_FALSE(acc.empty());
    ASSERT_EQ(acc.block().count(), N % bl_size);

    // build reference: compute block means manually, accumulate into var_acc
    auto data = make_data<sample_t>();
    auto acc_ref = make_empty<acc_t, sample_t>();
    auto block_mean_acc = make_empty<mean_acc<sample_t, A::varalg()>, sample_t>();
    for (int i = 0; i < N; ++i) {
        block_mean_acc << data[i];
        if ((i + 1) % bl_size == 0) {
            acc_ref << block_mean_acc.mean();
            block_mean_acc.reset();
        }
    }

    // the wrapped accumulator and block data should match the reference
    check_acc_equal(acc.accumulator(), acc_ref, tol);
    check_acc_equal(acc.block(), block_mean_acc, tol);
}

// Check that an autocorr_acc has the expected structure and data.
template <typename A>
void check_autocorr_acc(const A& acc, std::uint64_t c, double tol = 1e-10) {
    using acc_t = typename A::acc_type;
    using sample_t = typename acc_t::sample_type;

    // structural checks
    ASSERT_FALSE(acc.empty());
    ASSERT_EQ(acc.factor(), c);
    ASSERT_GE(acc.num_levels(), 2UL);

    // mean check
    check_near(acc.mean(), expected_mean<sample_t>(), tol);

    // for each level: verify B_l = c^l, N_l = floor(N / B_l), and that the accumulator matches a
    // reference built from block means
    auto data = make_data<sample_t>();
    auto const& bsizes = acc.block_sizes();
    for (std::size_t l = 0; l < acc.num_levels(); ++l) {
        ASSERT_EQ(bsizes[l], static_cast<std::uint64_t>(std::pow(c, l)));
        ASSERT_EQ(acc.count(l), N / bsizes[l]);
        ASSERT_EQ(acc.find_level(acc.count(l)), l);

        auto acc_ref = make_empty<acc_t, sample_t>();
        auto block_mean = make_empty<mean_acc<sample_t, A::varalg()>, sample_t>();
        for (int i = 0; i < N; ++i) {
            block_mean << data[i];
            if ((i + 1) % bsizes[l] == 0) {
                acc_ref << block_mean.mean();
                block_mean.reset();
            }
        }
        check_acc_equal(acc.accumulators()[l], acc_ref, tol);
    }
}

// Check that a batch_acc has the expected structure and data.
template <typename A>
void check_batch_acc(const A& acc, std::size_t m_b, double tol = 1e-10) {
    using sample_t = typename A::sample_type;
    using mean_acc_t = typename A::mean_acc_type;

    // compute expected batch_count by simulating merges
    std::uint64_t bcount = 1;
    auto cum = static_cast<std::uint64_t>(m_b);
    while (cum + m_b * bcount <= static_cast<std::uint64_t>(N)) {
        cum += m_b * bcount;
        bcount *= 2;
    }
    auto remaining = static_cast<std::uint64_t>(N) - cum;
    auto expected_full_batches = m_b + remaining / bcount;

    // structural checks
    ASSERT_EQ(acc.count(), N);
    ASSERT_FALSE(acc.empty());
    ASSERT_EQ(acc.batch_count(), bcount);
    ASSERT_EQ(acc.num_full_batches(), expected_full_batches);

    // build all 2*m_b reference batches in one pass over the data
    auto data = make_data<sample_t>();
    std::vector<mean_acc_t> ref_all(2 * m_b, make_empty<mean_acc_t, sample_t>());
    std::size_t bidx = 0;
    for (int i = 0; i < N; ++i) {
        ref_all[bidx] << data[i];
        if (ref_all[bidx].count() >= bcount && bidx + 1 < 2 * m_b) {
            ++bidx;
        }
    }

    // compare full and accumulating batches against the reference
    auto const& full = acc.batch_vector_full();
    auto const& accum = acc.batch_vector_accumulating();
    ASSERT_EQ(full.size(), m_b);
    ASSERT_EQ(accum.size(), m_b);
    for (std::size_t b = 0; b < m_b; ++b) {
        check_acc_equal(full[b], ref_all[b], tol);
        check_acc_equal(accum[b], ref_all[m_b + b], tol);
    }

    // for each merge factor c: merge reference batches, build reference var/covar accumulators from
    // the merged means, and compare against batch_acc results
    std::vector<mean_acc_t> ref_batches(ref_all.begin(), ref_all.begin() + static_cast<long>(expected_full_batches));
    for (std::size_t c = 1; c <= expected_full_batches; ++c) {
        auto n_merged = expected_full_batches / c;
        auto merged = acc.batches(c);
        ASSERT_EQ(merged.size(), n_merged);

        auto ref_merged = ref_batches;
        merge_batches(c, ref_merged);
        ASSERT_EQ(ref_merged.size(), n_merged);

        auto var_ref = make_empty<var_acc<sample_t, A::varalg()>, sample_t>();
        auto covar_ref = make_empty<covar_acc<sample_t, A::varalg()>, sample_t>();
        for (std::size_t b = 0; b < n_merged; ++b) {
            check_acc_equal(merged[b], ref_merged[b], tol);
            var_ref << ref_merged[b].mean();
            covar_ref << ref_merged[b].mean();
        }

        if (n_merged >= 2) {
            check_acc_equal(acc.var_accumulator(c), var_ref, tol);
            check_acc_equal(acc.covar_accumulator(c), covar_ref, tol);
        }
    }

    // mean_accumulator should aggregate all samples
    auto macc = acc.mean_accumulator();
    check_acc_results<sample_t>(macc, tol);
    check_near(acc.mean(), expected_mean<sample_t>(), tol);
}

} // namespace test_detail

#endif // SIMPLEMC_TESTS_ACCS_TEST_TRAITS_HPP
