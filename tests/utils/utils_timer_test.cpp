// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <simplemc/utils/timer.hpp>

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
    timer.interim();
    fmt::print("Time passed: {} ms\n", simplemc::time_passed(timer.start_time(), timer.interim_time(), millisec {}));
    timer.start();
    std::this_thread::sleep_for(millisec(100));
    fmt::print("Time passed: {} ms\n", timer.elapsed(millisec {}));
}
