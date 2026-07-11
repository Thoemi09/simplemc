// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/utils.hpp>

#include <thread>

int main() {
    // some convenient typedefs
    using millisec = simplemc::duration::millisec;
    using sec = simplemc::duration::sec;
    using min = simplemc::duration::min;

    // create a timer
    simplemc::timer timer;

    // sleep for 100 milliseconds and print the time passed in seconds
    fmt::println("Sleeping for 100 milliseconds...");
    timer.start();
    std::this_thread::sleep_for(millisec { 100 });
    timer.stop();
    fmt::println("Time passed: {} sec", simplemc::time_passed(timer.start_time(), timer.stop_time(), sec {}));
    fmt::println("");

    // sleep for 1 second and print the time passed in minutes
    fmt::println("Sleeping for 1 second...");
    timer.start();
    std::this_thread::sleep_for(sec { 1 });
    timer.stop();
    fmt::println("Time passed: {} min", simplemc::time_passed(timer.start_time(), timer.stop_time(), min {}));
}
