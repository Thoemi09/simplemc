#include <gtest/gtest.h>

#include <simplemc/config.hpp>
#include <simplemc/utils.hpp>
#include <simplemc/utils/fmt_complex.hpp>

#include <fmt/core.h>

// Test if the config header is generated correctly.
TEST(SimplemcUtils, ConfigHeader) {
    fmt::println("VERSION: {}", SIMPLEMC_VERSION);
#ifdef SIMPLEMC_WITH_H5
    fmt::println("SIMPLEMC_WITH_H5: True");
#else
    fmt::println("SIMPLEMC_WITH_H5: False");
#endif // SIMPLEMC_WITH_H5
#ifdef SIMPLEMC_USE_STD_RANGES
    fmt::println("SIMPLEMC_USE_STD_RANGES: True");
#else
    fmt::println("SIMPLEMC_USE_STD_RANGES: False");
#endif // SIMPLEMC_USE_STD_RANGES
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
