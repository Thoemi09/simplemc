/**
 * @file utils_test.cpp
 * @brief Unit tests for utils library.
 */

#include "simplemc/utils/generic_error.hpp"
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