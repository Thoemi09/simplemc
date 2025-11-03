#include <fmt/base.h>
#include <gtest/gtest.h>
#include <simplemc/config.hpp>
#include <simplemc/utils.hpp>
#include <simplemc/utils/fmt_complex.hpp>

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

// Test file IO with raw functions.
TEST(SimplemcUtils, FileIORaw) {
    auto fp = simplemc::detail::open_file_raw("test_file_raw.txt", "w");
    fmt::print(fp, "This is test file #{}\n", 1);
    simplemc::detail::close_file_raw(fp, "test_file_raw.txt");
}

// Test file IO with RAII wrapper (recommended API).
TEST(SimplemcUtils, FileIO) {
    auto file = simplemc::open_file("test_file.txt", "w");
    fmt::print(file, "This is test file #{}\n", 1);
    ASSERT_EQ(file.name(), "test_file.txt");
}

// Test file_handle direct construction and advanced features.
TEST(SimplemcUtils, FileHandleRAII) {
    {
        simplemc::file_handle file("test_file_raii.txt", "w");
        fmt::print(file.get(), "This is RAII test file #{}\n", 2);
        ASSERT_EQ(file.name(), "test_file_raii.txt");
    }

    // test move semantics
    simplemc::file_handle file1("test_file_move.txt", "w");
    fmt::print(file1.get(), "Line 1\n");

    simplemc::file_handle file2(std::move(file1));
    ASSERT_EQ(file1.get(), nullptr);
    ASSERT_NE(file2.get(), nullptr);
    ASSERT_EQ(file2.name(), "test_file_move.txt");
    fmt::print(file2, "Line 2\n");

    // test explicit close (and verify no double-close in destructor)
    file2.close();
    ASSERT_EQ(file2.get(), nullptr);
}
