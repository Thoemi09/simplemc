@page tutorials Tutorials

[TOC]

| Tutorial | Description |
|---------|-------------|
| @ref tut_utils_1 | Measure the runtime of your program using simplemc::timer |
| @ref tut_utils_2 | Convert multi-dimensional to flat indices using simplemc::flat_index |
| @ref tut_utils_3 | Formatted output with **fmt** |
| @ref tut_mpi_1 | Introduction to simplemc::mpi::environment and simplemc::mpi::communicator |
| @ref tut_mpi_2 | Effect of the `abort_on_exception` flag on the simplemc::mpi::environment |
| @ref tut_mpi_3 | How to use various @ref simplemc-mpi-coll |
| @ref tut_mpi_4 | Make your custom type compatible with @ref simplemc-mpi |
| @ref tut_random_1 | Approximating PDFs with histograms from @ref simplemc-random-samples |
| @ref tut_grids_1 | Showcase the different @ref simplemc-grids-1d |
| @ref tut_grids_2 | Showcase @ref simplemc-grids-nd |
| @ref tut_numeric_1 | Working with @ref simplemc-numeric-lattices |

@section tut_compilation Compiling the tutorials

All tutorials have been compiled on a MacBook Pro with an Apple M2 Max chip and clang++ v19.1.6
together with cmake v3.31.3.

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
