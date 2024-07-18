#include <gtest/gtest.h>

#include <simplemc/utils/timer.hpp>

#include <fmt/core.h>

#include <thread>

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
