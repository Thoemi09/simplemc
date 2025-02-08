#ifndef SIMPLEMC_TESTS_TEST_UTILS_HPP
#define SIMPLEMC_TESTS_TEST_UTILS_HPP

#include <gtest/gtest.h>

#include <range/v3/all.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

// Check for nearness.
template <typename T1, typename T2>
inline void check_near(T1 lhs, T2 rhs, double eps = 1e-14) {
    ASSERT_NEAR(lhs, rhs, eps);
}

// Check complex numbers for nearness
template <typename T1, typename T2>
inline void check_near(std::complex<T1> lhs, std::complex<T2> rhs, double eps = 1e-14) {
    ASSERT_NEAR(std::real(lhs), std::real(rhs), eps);
    ASSERT_NEAR(std::imag(lhs), std::imag(rhs), eps);
}

// Check for equality.
template <typename T1, typename T2>
inline void check_equal(T1 lhs, T2 rhs) {
    if constexpr (std::integral<T1> && std::integral<T2>) {
        ASSERT_EQ(lhs, rhs);
    } else {
        ASSERT_DOUBLE_EQ(lhs, rhs);
    }
}

// Check complex numbers for double equality.
template <typename T1, typename T2>
inline void check_equal(std::complex<T1> lhs, std::complex<T2> rhs) {
    ASSERT_DOUBLE_EQ(std::real(lhs), std::real(rhs));
    ASSERT_DOUBLE_EQ(std::imag(lhs), std::imag(rhs));
}

// Check two ranges for nearness.
void check_range_near(ranges::input_range auto&& rg, ranges::input_range auto&& exp_rg, double eps = 1e-14) {
    ASSERT_EQ(ranges::size(rg), ranges::size(exp_rg));
    for (auto&& [x, y] : ranges::views::zip(rg, exp_rg)) {
        check_near(x, y, eps);
    }
}

// Check two ranges for equality.
void check_range_equal(ranges::input_range auto&& rg, ranges::input_range auto&& exp_rg) {
    ASSERT_EQ(ranges::size(rg), ranges::size(exp_rg));
    for (auto&& [x, y] : ranges::views::zip(rg, exp_rg)) {
        check_equal(x, y);
    }
}

// Check if a value is NaN.
void check_isnan(auto val) {
    if constexpr (std::is_same_v<std::remove_reference_t<decltype(val)>, double>) {
        ASSERT_TRUE(std::isnan(val));
    } else {
        ASSERT_TRUE(std::isnan(val.real()));
        ASSERT_TRUE(std::isnan(val.imag()));
    }
}

// Simple histogram class on the interval [0, 1] for testing.
class histogram01 {
public:
    // Constructor.
    explicit histogram01(int nbins) : nbins_(nbins), step_(1.0 / nbins), hist_(nbins, 0.0) {}

    // Add a value to the histogram.
    void add(double value);

    // Print the histogram.
    void print() const;

    // Check that the histogram is uniform within some tolerance.
    [[nodiscard]] bool check_uniform(double tol) const;

private:
    long nsamples_ { 0 };
    int nbins_;
    double step_;
    std::vector<double> hist_;
};

#endif // SIMPLEMC_TESTS_TEST_UTILS_HPP
