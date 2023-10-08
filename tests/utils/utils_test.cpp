/**
 * @file utils_test.cpp
 * @brief Unit tests for utils library.
 */

#include <simplemc/utils/format.h>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/config.h>

#include <fmt/format.h>
#include <gtest/gtest.h>

// Test if the config header is generated correctly.
TEST(SimplemcUtils, ConfigHeader) {
    fmt::print("VERSION: {}\n", SIMPLEMC_VERSION);
}

// Test generic_error exception.
TEST(SimplemcUtils, GenericError) {
    try {
        throw simplemc::generic_error("generic_error", "Test message", "test_function");
    } catch (const simplemc::generic_error& e) {
        fmt::print("Caught exception: {}\n", e.what());
    }
}

// Test simplemc_exception exception.
TEST(SimplemcUtils, SimplemcException) {
    try {
        throw simplemc::simplemc_exception("Test message");
    } catch (const simplemc::simplemc_exception& e) {
        fmt::print("Caught exception: {}\n", e.what());
    }
}

// Test specialized formatter for std::complex.
TEST(SimplemcUtils, ComplexFormatter) {
    std::complex<double> z(1.0, 2.0);
    fmt::print("z = {}\n", z);
}