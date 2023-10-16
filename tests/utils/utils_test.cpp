/**
 * @file utils_test.cpp
 * @brief Unit tests for utils library.
 */
 
#include "../test_utils.hpp"

#include <simplemc/config.hpp>
#include <simplemc/utils/file_io.hpp>
#include <simplemc/utils/format.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/format.h>

#include <thread>

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