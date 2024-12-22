@page tutorials Tutorials

[TOC]

@section tut_utils simplemc-utils

- @ref tut_utils_1
- @ref tut_utils_2
- @ref tut_utils_3

@section tut_mpi simplemc-mpi

- @ref tut_mpi_1
- @ref tut_mpi_2
- @ref tut_mpi_3

@section tut_compilation Compiling the tutorials

All tutorials have been compiled on a MacBook Pro with an Apple M2 Max chip and clang++ v19.1.2
together with cmake v3.30.5.

Furhtermore, the following dependencies have been used:
- [open-mpi](https://www.open-mpi.org/) 5.0.1 (installed with brew)

Assuming that the actual tutorial code is in a file `main.cpp`, the following generic `CMakeLists.txt`
should work for all tutorials:

```cmake
cmake_minimum_required(VERSION 3.24)
project(example CXX)

# set required standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# fetch simplemc from github
include(FetchContent)
FetchContent_Declare(
  simplemc
  GIT_REPOSITORY git@github.com:Thoemi09/simplemc.git
  GIT_TAG main
)
FetchContent_MakeAvailable(simplemc)

# build the example
add_executable(ex main.cpp)
target_link_libraries(ex simplemc::simplemc)
```
