#include <gtest/gtest.h>

#include <simplemc/config.hpp>
#include <simplemc/utils.hpp>
#include <simplemc/utils/fmt_complex.hpp>

#include <fmt/core.h>

// Test if the config header is generated correctly.
TEST(SimplemcUtils, ConfigHeader) {
    fmt::print("VERSION: {}\n", SIMPLEMC_VERSION);
#ifdef SIMPLEMC_WITH_HIGHFIVE
    fmt::print("SIMPLEMC_WITH_HIGHFIVE: True\n");
#else
    fmt::print("SIMPLEMC_WITH_HIGHFIVE: False\n");
#endif // SIMPLEMC_WITH_HIGHFIVE
}

// Test concepts.
TEST(SimplemcUtils, Concepts) {
    ASSERT_TRUE(simplemc::integer_only<int>);
    ASSERT_FALSE(simplemc::integer_only<char>);
    ASSERT_TRUE(simplemc::integer_only<std::uint64_t>);
    ASSERT_FALSE(simplemc::integer_only<std::int8_t>);
    ASSERT_TRUE(simplemc::nd_order<simplemc::row_major>);
    ASSERT_TRUE(simplemc::nd_order<simplemc::column_major>);
    ASSERT_FALSE(simplemc::nd_order<int>);
    ASSERT_TRUE(simplemc::double_or_complex<double>);
    ASSERT_TRUE(simplemc::double_or_complex<std::complex<double>>);
    ASSERT_FALSE(simplemc::double_or_complex<float>);
    ASSERT_FALSE(simplemc::double_or_complex<std::complex<float>>);
}

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

// Test specialized formatter for std::complex.
TEST(SimplemcUtils, ComplexFormatter) {
    std::complex<double> z(1.0003, 2.00041291823);
    fmt::print("z = {:^30.15}\n", z);
}

// Test file IO.
TEST(SimplemcUtils, FileIO) {
    auto fp = simplemc::open_file("test_file.txt", "w");
    fmt::print(fp, "This is test file #{}\n", 1);
    simplemc::close_file(fp);
}
