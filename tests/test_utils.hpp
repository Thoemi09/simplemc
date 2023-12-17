/**
 * @file test_utils.hpp
 * @brief Some helper utilities for testing.
 */

#ifndef SIMPLEMC_TESTS_TEST_UTILS_HPP
#define SIMPLEMC_TESTS_TEST_UTILS_HPP

#include <gtest/gtest.h>
#include <simplemc/json/json.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip.hpp>

// Check JSON serialization of an object.
template <typename T1, typename T2>
void check_json(const T1& orig, T2& copy) {
    nlohmann::json j;
    j["orig"] = orig;
    j.at("orig").get_to(copy);
    j["copy"] = copy;
    ASSERT_EQ(j["orig"], j["copy"]);
}

// Check JSON serialization of an object (needs to be default constructible).
template <typename T>
void check_json(const T& orig) {
    T copy;
    check_json(orig, copy);
}

// Check complex numbers.
inline void check_complex(std::complex<double> lhs, std::complex<double> rhs) {
    ASSERT_DOUBLE_EQ(std::real(lhs), std::real(rhs));
    ASSERT_DOUBLE_EQ(std::imag(lhs), std::imag(rhs));
}

// Check two ranges for nearness.
void check_range_near(ranges::input_range auto&& rg, ranges::input_range auto&& exp_rg, double eps = 1e-14) {
    for (auto&& [x, y] : ranges::views::zip(rg, exp_rg)) {
        ASSERT_NEAR(x, y, eps);
    }
}

// Check two ranges for equality.
void check_range_equal(ranges::input_range auto&& rg, ranges::input_range auto&& exp_rg) {
    for (auto&& [x, y] : ranges::views::zip(rg, exp_rg)) {
        ASSERT_EQ(x, y);
    }
}

#endif // SIMPLEMC_TESTS_TEST_UTILS_HPP
