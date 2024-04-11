@page integration Integration in C++ projects

[TOC]

**simplemc** is a compiled library.
You can either build/install it beforehand (see @ref installation) and then link against it or let CMake fetch and
build it directly as part of your project.

To use **simplemc** in your own C++ code, you simply have to include the relevant header files. For example:

```cpp
#include <simplemc/mpi.hpp>

// use simplemc::mpi
```

In the following, we describe some common ways to achieve this (with special focus on CMake).

@section cmake CMake

@subsection fetch FetchContent

If you use [CMake](https://cmake.org/) to build your source code, it is recommended to fetch the source code directly from the
[Github repository](https://github.com/Thoemi09/simplemc) using CMake's
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) module:

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_project CXX)

# fetch from github
include(FetchContent)
FetchContent_Declare(
  simpelmc
  GIT_REPOSITORY git@github.com:Thoemi09/simplemc.git
  GIT_TAG main
)
FetchContent_MakeAvailable(simplemc)

# declare a target and link to simplemc
add_executable(my_executable main.cpp)
target_link_libraries(my_executable simplemc::simplemc)
```

To have more control over how **simplemc** is built and how dependencies are fetched, please take a look at the @ref cmake_options.

@subsection find_package find_package

If you have already installed **simplemc** on your system by following the instructions from the @ref installation page,
you can also make use of CMake's [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command.
This has the advantage that you don't need to download anything, i.e. no internet connection is required.

Let's assume that **simplemc** has been installed to `path_to_install_dir`.
Then linking your project to **simplemc** with CMake is as easy as

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_project CXX)

# find simplemc
find_package(simpelmc REQUIRED CONFIG)

# declare a target and link to simplemc
add_executable(my_executable main.cpp)
target_link_libraries(my_executable simplemc::simplemc)
```

In case, CMake cannot find the package, you might have to tell it where to look for the `simplemc-config.cmake` file
by setting the variable `simplemc_DIR` to `path_to_install_dir/lib/cmake/simplemc`.

@subsection add_sub add_subdirectory

You can also integrate **simplemc** into our CMake project by placing the entire source tree in a subdirectory and call `add_subdirectory()`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_project CXX)

# add simplemc subdirectory
add_subdirectory(deps/simplemc)

# declare a target and link to simplemc
add_executable(my_executable main.cpp)
target_link_libraries(my_executable simplemc::simplemc)
```

Here, it is assumed that the **simplemc** source tree is in a subdirectory `deps/simplemc` relative to your `CMakeLists.txt` file.
