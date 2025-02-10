#include <fmt/format.h>
#include <gtest/gtest.h>
#include <simplemc/utils/generic_error.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

// Test generic_error exception.
TEST(SimplemcUtils, GenericError) {
    try {
        throw simplemc::generic_error("generic_error", "Test message", "test_function");
    } catch (const simplemc::generic_error& e) {
        fmt::print("Caught exception: {}\n", e.what());
    }

    try {
        throw simplemc::generic_error("generic_error", fmt::format("Test message {}", 2));
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
