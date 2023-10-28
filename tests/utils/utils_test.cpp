/**
 * @file utils_test.cpp
 * @brief Unit tests for utils library.
 */

#include <gtest/gtest.h>

#include <simplemc/config.hpp>
#include <simplemc/utils/concepts.hpp>
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

// Test if the config header is generated correctly.
TEST(SimplemcUtils, Concepts) {
    ASSERT_TRUE(simplemc::integer_only<int>);
    ASSERT_FALSE(simplemc::integer_only<char>);
    ASSERT_TRUE(simplemc::integer_only<std::uint64_t>);
    ASSERT_FALSE(simplemc::integer_only<std::int8_t>);
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
    constexpr std::array<int, 2> shape_arr_cepr { 2, 3 };
    constexpr auto size_cepr = size_from_shape_array(shape_arr_cepr);
    ASSERT_EQ(size_cepr, 6);
    std::array<long, 2> shape_arr { 2, 3 };
    auto size = size_from_shape_array(shape_arr);
    ASSERT_EQ(size, 6);
}

// Test constexpr indexing with arrays.
TEST(SimplemcUtils, ConstexprIndexing) {
    using namespace simplemc;
    constexpr std::array<int, 3> shape { 2, 3, 4 };
    constexpr auto size = size_from_shape_array(shape);
    ASSERT_EQ(size, 24);
    constexpr std::array<int, 3> exp_idxs1 { 0, 0, 0 };
    constexpr int exp_fidx1 = 0;
    constexpr auto cfidx1 = flat_index_array(exp_idxs1, shape);
    constexpr auto cidxs1 = multi_index_array(exp_fidx1, shape);
    constexpr auto rfidx1 = flat_index_array(exp_idxs1, shape, row_major {});
    constexpr auto ridxs1 = multi_index_array(exp_fidx1, shape, row_major {});
    ASSERT_EQ(cfidx1, exp_fidx1);
    ASSERT_EQ(cidxs1, exp_idxs1);
    ASSERT_EQ(rfidx1, exp_fidx1);
    ASSERT_EQ(ridxs1, exp_idxs1);
    constexpr std::array<int, 3> exp_idxs2 { 1, 2, 3 };
    constexpr int exp_fidx2 = 23;
    constexpr auto cfidx2 = flat_index_array(exp_idxs2, shape);
    constexpr auto cidxs2 = multi_index_array(exp_fidx2, shape);
    constexpr auto rfidx2 = flat_index_array(exp_idxs2, shape, row_major {});
    constexpr auto ridxs2 = multi_index_array(exp_fidx2, shape, row_major {});
    ASSERT_EQ(cfidx2, exp_fidx2);
    ASSERT_EQ(cidxs2, exp_idxs2);
    ASSERT_EQ(rfidx2, exp_fidx2);
    ASSERT_EQ(ridxs2, exp_idxs2);
    constexpr std::array<int, 3> exp_idxs3 { 1, 1, 1 };
    constexpr int exp_cfidx3 = 9;
    constexpr int exp_rfidx3 = 17;
    constexpr auto cfidx3 = flat_index_array(exp_idxs3, shape);
    constexpr auto cidxs3 = multi_index_array(exp_cfidx3, shape);
    constexpr auto rfidx3 = flat_index_array(exp_idxs3, shape, row_major {});
    constexpr auto ridxs3 = multi_index_array(exp_rfidx3, shape, row_major {});
    ASSERT_EQ(cfidx3, exp_cfidx3);
    ASSERT_EQ(cidxs3, exp_idxs3);
    ASSERT_EQ(rfidx3, exp_rfidx3);
    ASSERT_EQ(ridxs3, exp_idxs3);
}

// Test indexing with vectors.
TEST(SimplemcUtils, Indexing) {
    using namespace simplemc;
    std::vector<long> shape { 2, 3, 4 };
    auto size = size_from_shape(shape);
    ASSERT_EQ(size, 24);
    std::vector<long> exp_idxs { 0, 0, 0 };
    int exp_fidx = 0;
    ASSERT_EQ(flat_index(exp_idxs, shape), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape), exp_idxs);
    ASSERT_EQ(flat_index(exp_idxs, shape, row_major {}), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape, row_major {}), exp_idxs);
    exp_idxs = { 1, 2, 3 };
    exp_fidx = 23;
    ASSERT_EQ(flat_index(exp_idxs, shape), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape), exp_idxs);
    ASSERT_EQ(flat_index(exp_idxs, shape, row_major {}), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape, row_major {}), exp_idxs);
    exp_idxs = { 1, 1, 1 };
    exp_fidx = 9;
    ASSERT_EQ(flat_index(exp_idxs, shape), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape), exp_idxs);
    exp_fidx = 17;
    ASSERT_EQ(flat_index(exp_idxs, shape, row_major {}), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape, row_major {}), exp_idxs);
}
