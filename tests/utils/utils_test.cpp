/**
 * @file utils_test.cpp
 * @brief Unit tests for utils library.
 */
 
#include <gtest/gtest.h>

#include <simplemc/config.hpp>
#include <simplemc/utils/file_io.hpp>
#include <simplemc/utils/format.hpp>
#include <simplemc/utils/indexing.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/ranges.h>
#include <range/v3/all.hpp>

#include <thread>
#include <vector>

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
    std::complex<double> z(1.0003, 2.00041291823);
    fmt::print("z = {:^30.15}\n", z);
}

// Test file IO.
TEST(SimplemcUtils, FileIO) {
    auto fp = simplemc::open_file("test_file.txt", "w");
    fmt::print(fp, "This is test file #{}\n", 1);
    simplemc::close_file(fp);
}

// Test timer.
TEST(SimplemcUtils, Timer) {
    using millisec = simplemc::duration::millisec;
    simplemc::timer timer;
    timer.start();
    std::this_thread::sleep_for(millisec(100));
    timer.stop();
    fmt::print("Time passed: {} ms\n", simplemc::time_passed(timer.start_time(), timer.stop_time(), millisec {}));
    timer.start();
    std::this_thread::sleep_for(millisec(100));
    timer.stop();
    fmt::print("Time passed: {} ms\n", simplemc::time_passed(timer.start_time(), timer.stop_time(), millisec {}));
    timer.start();
    std::this_thread::sleep_for(millisec(100));
    timer.stop();
    fmt::print("Time passed: {} ms\n", simplemc::time_passed(timer.start_time(), timer.stop_time(), millisec {}));
}

// Test calculating the size from a shape.
TEST(SimplemcUtils, SizeFromShape) {
    using namespace simplemc;
    std::vector<int> shape;
    ASSERT_EQ(size_from_shape(shape), 0);
    shape = { 2, 3, 4 };
    ASSERT_EQ(size_from_shape(shape), 24);
    ASSERT_EQ(size_from_shape(ranges::views::repeat_n(5, 3)), 5 * 5 * 5);
    ASSERT_EQ(size_from_shape(ranges::views::repeat_n(5, 0)), 0);
    constexpr std::array<int, 2> shape_arr_cepr { 2, 3 };
    constexpr auto size_cepr = size_from_shape(shape_arr_cepr);
    ASSERT_EQ(size_cepr, 6);
    std::array<long, 2> shape_arr { 2, 3 };
    auto size = size_from_shape(shape_arr);
    ASSERT_EQ(size, 6);
}

// Test converting a flat index to a multi-dimensional index.
TEST(SimplemcUtils, MultiIndex) {
    using namespace simplemc;
    constexpr std::array<int, 2> shape_arr_cepr { 2, 3 };
    constexpr auto idxs_cepr = multi_index(4, shape_arr_cepr);
    fmt::print("idxs_cepr = {}\n", idxs_cepr);
    std::array<long, 2> shape_arr { 2, 3 };
    auto idxs = multi_index(4l, shape_arr, row_major {});
    fmt::print("idxs = {}\n", idxs);
}