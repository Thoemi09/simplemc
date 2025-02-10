@page tut_utils_1 Utilities Tutorial 1: Runtime measurement

[TOC]

In this tutorial, we show how to measure the runtime of a program using the simplemc::timer class from
the @ref simplemc-utils library.

@section tut_utils_1_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <simplemc/utils.hpp>

#include <thread>

int main() {
    // code snippets go here
}
```

Let's start with some convenient typedefs:

```cpp
// some convenient typedefs
using millisec = simplemc::duration::millisec;
using sec = simplemc::duration::sec;
using min = simplemc::duration::min;
```

They are taken from simplemc::duration and allow us straightforwardly output time differences in
milliseconds, seconds and minutes without any effort.

Then we create an instance of the simplemc::timer class:

```cpp
// create a timer
simplemc::timer timer;
```

We will now use `std::this_thread::sleep_for` to sleep for a certain amount of time and check this
amount with our timer object.

First let's sleep for 100 milliseconds:

```cpp
// sleep for 100 milliseconds and print the time passed in seconds
fmt::println("Sleeping for 100 milliseconds...");
timer.start();
std::this_thread::sleep_for(millisec { 100 });
timer.stop();
fmt::println("Time passed: {} sec", simplemc::time_passed(timer.start_time(), timer.stop_time(), sec {}));
fmt::println("");
```

Output:

```
Sleeping for 100 milliseconds...
Time passed: 0.105088917 sec
```

Here, we use the simplemc::timer::start and simplemc::timer::stop methods to record the time points
before we go to sleep and after we wake up, respectively.

The recorded time points are then forwarded to simplemc::time_passed to get the time difference.
Note how we use `millisec { 100 }` in the sleep-function and `sec {}` when printing the output.

Let's try this again with sleeping for 1 second and output the time in minutes:

```cpp
// sleep for 1 second and print the time passed in minutes
fmt::println("Sleeping for 1 second...");
timer.start();
std::this_thread::sleep_for(sec { 1 });
timer.stop();
fmt::println("Time passed: {} min", simplemc::time_passed(timer.start_time(), timer.stop_time(), min {}));
```

Output:

```
Sleeping for 1 second...
Time passed: 0.01667623125 min
```

@section tut_utils_1_code Full code

```cpp
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
```
