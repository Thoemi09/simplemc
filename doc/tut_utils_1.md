@page tut_utils_1 Utilities Tutorial 1: Runtime measurement

[TOC]

In this tutorial, we show how to measure the runtime of a program using the simplemc::timer class from the
@ref simplemc-utils library.

```cpp
#include <simplemc/utils.hpp>
#include <thread>

int main() {
    // some convenient typedefs
    using millisec = simplemc::duration::millisec;
    using sec = simplemc::duration::sec;
    using min = simplemc::duration::min;

    // create a timer
    simplemc::timer timer;

    // sleep for 100 millisecond and print the time passed in milliseconds
    fmt::print("Sleeping for 1 second...\n");
    timer.start();
    std::this_thread::sleep_for(millisec(100));
    timer.stop();
    fmt::print("Time passed: {} ms\n\n", simplemc::time_passed(timer.start_time(), timer.stop_time(), sec {}));

    // sleep for 1 second and print the time passed in minutes
    fmt::print("Sleeping for 1 second...\n");
    timer.start();
    std::this_thread::sleep_for(sec(1));
    timer.stop();
    fmt::print("Time passed: {} min\n", simplemc::time_passed(timer.start_time(), timer.stop_time(), min {}));
}
```

Output:

```
Sleeping for 1 second...
Time passed: 0.105040333 ms

Sleeping for 1 second...
Time passed: 0.01667623125 min
```
