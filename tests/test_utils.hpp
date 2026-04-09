#ifndef SIMPLEMC_TESTS_TEST_UTILS_HPP
#define SIMPLEMC_TESTS_TEST_UTILS_HPP

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

// Check scalars for nearness.
template <typename T1, typename T2>
    requires(!simplemc::ranges::input_range<T1> && !simplemc::eigen_matrix<std::remove_cvref_t<T1>>)
void check_near(T1 lhs, T2 rhs, double eps = 1e-14) {
    ASSERT_NEAR(lhs, rhs, eps);
}

// Check complex numbers for nearness.
template <typename T1, typename T2>
void check_near(std::complex<T1> lhs, std::complex<T2> rhs, double eps = 1e-14) {
    ASSERT_NEAR(std::real(lhs), std::real(rhs), eps);
    ASSERT_NEAR(std::imag(lhs), std::imag(rhs), eps);
}

// Check two ranges for element-wise nearness. Recursively dispatches to the scalar or nested-range
// overload.
void check_near(
    simplemc::ranges::input_range auto&& rg, simplemc::ranges::input_range auto&& exp_rg, double eps = 1e-14) {
    ASSERT_EQ(simplemc::ranges::size(rg), simplemc::ranges::size(exp_rg));
    for (auto&& [x, y] : simplemc::ranges::views::zip(rg, exp_rg)) {
        check_near(x, y, eps);
    }
}

// Check two Eigen matrices for element-wise nearness (for types that are not input_range, i.e. 2D
// matrices as opposed to column vectors).
template <typename M1, typename M2>
    requires(simplemc::eigen_matrix<std::remove_cvref_t<M1>> && simplemc::eigen_matrix<std::remove_cvref_t<M2>> &&
        !simplemc::ranges::input_range<M1>)
void check_near(const M1& lhs, const M2& rhs, double eps = 1e-14) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            check_near(lhs(i, j), rhs(i, j), eps);
        }
    }
}

// Check scalars for equality.
template <typename T1, typename T2>
    requires(!simplemc::ranges::input_range<T1>)
void check_equal(T1 lhs, T2 rhs) {
    if constexpr (std::integral<T1> && std::integral<T2>) {
        ASSERT_EQ(lhs, rhs);
    } else {
        ASSERT_DOUBLE_EQ(lhs, rhs);
    }
}

// Check complex numbers for double equality.
template <typename T1, typename T2>
void check_equal(std::complex<T1> lhs, std::complex<T2> rhs) {
    ASSERT_DOUBLE_EQ(std::real(lhs), std::real(rhs));
    ASSERT_DOUBLE_EQ(std::imag(lhs), std::imag(rhs));
}

// Check two ranges for element-wise equality.
void check_equal(simplemc::ranges::input_range auto&& rg, simplemc::ranges::input_range auto&& exp_rg) {
    ASSERT_EQ(simplemc::ranges::size(rg), simplemc::ranges::size(exp_rg));
    for (auto&& [x, y] : simplemc::ranges::views::zip(rg, exp_rg)) {
        check_equal(x, y);
    }
}

// Check if a scalar value is NaN.
template <typename T>
    requires(!simplemc::ranges::input_range<T> && !simplemc::eigen_matrix<std::remove_cvref_t<T>>)
void check_isnan(T val) {
    if constexpr (std::is_same_v<std::remove_reference_t<decltype(val)>, double>) {
        ASSERT_TRUE(std::isnan(val));
    } else {
        ASSERT_TRUE(std::isnan(val.real()));
        ASSERT_TRUE(std::isnan(val.imag()));
    }
}

// Check that all elements of a range are NaN.
void check_isnan(simplemc::ranges::input_range auto&& rg) {
    for (auto&& x : rg) {
        check_isnan(x);
    }
}

// Check that all elements of an Eigen matrix are NaN (for types that are not input_range, i.e. 2D
// matrices as opposed to column vectors).
template <typename M>
    requires(simplemc::eigen_matrix<std::remove_cvref_t<M>> && !simplemc::ranges::input_range<M>)
void check_isnan(const M& mat) {
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            check_isnan(mat(i, j));
        }
    }
}

// Check an empty accumulator (shared with basic and advanced accumulator tests).
template <typename A>
void check_acc_empty(const A& acc) {
    ASSERT_EQ(acc.count(), 0);
    ASSERT_TRUE(acc.empty());
    check_isnan(acc.mean());
}

// Simple histogram class on the interval [a, b] for testing.
struct histogram {
    // Constructor.
    histogram(double a, double b, int nbins) : a(a), b(b), nbins(nbins), step((b - a) / nbins), data(nbins, 0.0) {}

    // Add a value to the histogram.
    void add(double value) {
        nsamples += 1;
        if (value < a || value >= b) {
            return;
        }
        int idx = static_cast<int>((value - a) / step);
        data[idx] += 1.0;
    }

    // Print the histogram + a reference function.
    void print(auto f = [](auto /* x */) { return 0; }) const {
        const auto norm = static_cast<double>(nsamples) * step;
        fmt::println("{:<15}{:<15}{:<15}", "x", "h(x)", "f(x)");
        for (int i = 0; i < nbins; ++i) {
            const auto x = a + step * (i + 0.5);
            fmt::println("{:<15.8f}{:<15.8f}{:<15.8f}", x, data[i] / norm, f(x));
        }
    }

    // Check that the histogram is close to a reference function within a tolerance.
    [[nodiscard]] bool check(auto f, double tol) const {
        const auto norm = static_cast<double>(nsamples) * step;
        for (int i = 0; i < nbins; ++i) {
            const auto x = a + step * (i + 0.5);
            const double fx = f(x);
            if (data[i] / norm < fx * (1 - tol) || data[i] / norm > fx * (1 + tol)) {
                fmt::println("Histogram check failed: {} < {} || {} > {}", data[i] / norm, fx * (1 - tol),
                    data[i] / norm, fx * (1 + tol));
                return false;
            }
        }
        return true;
    }

    double a;
    double b;
    int nbins;
    double step;
    std::vector<double> data;
    long nsamples { 0 };
};

#endif // SIMPLEMC_TESTS_TEST_UTILS_HPP
